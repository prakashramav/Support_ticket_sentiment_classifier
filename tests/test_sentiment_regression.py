"""
Sentiment Classification Regression, Label Mapping, Calibration & Reliability Tests
Support-Ticket Sentiment Classifier
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import backend.main as backend_main
from backend.app.services.model_loader import ModelService, preprocess_input_text


NEGATIVE_EXAMPLES = [
    "The movie is very bad, I don't like it anymore, the characters are not good.",
    "The movie was terrible and very disappointing.",
    "I hated the service. It was extremely poor.",
    "The product is bad and completely useless.",
    "I am unhappy with this experience.",
    "The movie is very bad, I don't like it, and the characters are not good.",
]

POSITIVE_EXAMPLES = [
    "The movie was excellent and wonderful.",
    "I absolutely loved the service.",
    "The product is amazing and very useful.",
    "I am extremely happy with this experience.",
    "The performance was outstanding.",
]


class TestSentimentRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ModelService(lazy=False, strict=True)

    def test_dataset_label_encoding_and_no_leakage(self):
        train_df = pd.read_csv(PROJECT_ROOT / "data" / "sst2" / "train.csv")
        val_df = pd.read_csv(PROJECT_ROOT / "data" / "sst2" / "validation.csv")

        self.assertEqual(set(train_df["label"].unique()), {0, 1})
        self.assertEqual(set(val_df["label"].unique()), {0, 1})

        train_norm = set(train_df["sentence"].astype(str).str.strip().str.lower())
        val_norm = set(val_df["sentence"].astype(str).str.strip().str.lower())
        self.assertEqual(len(train_norm & val_norm), 0)

    def test_bert_label_mapping_and_calibration_structure(self):
        self.assertEqual(self.service.bert_id2label, {0: "NEGATIVE", 1: "POSITIVE"})

        res = self.service.predict_bert(
            "The movie is very bad, I don't like it anymore, the characters are not good."
        )
        self.assertEqual(res["sentiment"], "NEGATIVE")
        self.assertEqual(res["rawClassId"], 0)
        self.assertEqual(res["rawLabel"], "NEGATIVE")
        self.assertEqual(len(res["rawLogits"]), 2)
        self.assertGreater(res["probabilities"]["NEGATIVE"], res["probabilities"]["POSITIVE"])
        self.assertGreater(
            res["calibratedProbabilities"]["NEGATIVE"],
            res["calibratedProbabilities"]["POSITIVE"],
        )
        # With temperature scaling (T > 1), calibrated confidence tempers overconfidence vs raw probability
        self.assertLessEqual(res["calibratedConfidence"], res["modelProbability"] + 1e-6)
        self.assertEqual(res["confidence"], res["calibratedConfidence"])
        self.assertEqual(res["calibration"]["method"], "temperature_scaling")
        self.assertEqual(res["source"], "live-model")

    def test_tfidf_class_mapping_and_vocab_consistency(self):
        self.assertEqual(self.service.tfidf_col_to_label, {0: "NEGATIVE", 1: "POSITIVE"})
        vocab = self.service.tfidf_pipeline[0].vocabulary_
        self.assertIn("unhappy", vocab)
        self.assertEqual(list(self.service.tfidf_pipeline.classes_), [0, 1])

        res = self.service.predict_tfidf("I am unhappy with this experience.")
        self.assertEqual(res["sentiment"], "NEGATIVE")
        self.assertEqual(res["rawClassId"], 0)
        self.assertGreater(res["probabilities"]["NEGATIVE"], res["probabilities"]["POSITIVE"])

    def test_negative_regression_suite_both_models(self):
        for text in NEGATIVE_EXAMPLES:
            with self.subTest(text=text):
                bert_res = self.service.predict_bert(text)
                tfidf_res = self.service.predict_tfidf(text)
                self.assertEqual(bert_res["sentiment"], "NEGATIVE", f"BERT failed on: {text}")
                self.assertEqual(tfidf_res["sentiment"], "NEGATIVE", f"TF-IDF failed on: {text}")

    def test_positive_regression_suite_both_models(self):
        for text in POSITIVE_EXAMPLES:
            with self.subTest(text=text):
                bert_res = self.service.predict_bert(text)
                tfidf_res = self.service.predict_tfidf(text)
                self.assertEqual(bert_res["sentiment"], "POSITIVE", f"BERT failed on: {text}")
                self.assertEqual(tfidf_res["sentiment"], "POSITIVE", f"TF-IDF failed on: {text}")

    def test_contrastive_negation_bert(self):
        neg = self.service.predict_bert("Not good at all, I won't buy this again.")
        pos = self.service.predict_bert("Not bad at all, actually quite helpful!")
        self.assertEqual(neg["sentiment"], "NEGATIVE")
        self.assertEqual(pos["sentiment"], "POSITIVE")

    def test_input_validation_guards(self):
        for bad_input in ["", "   \n\t ", "!!!", "...", "---???"]:
            with self.assertRaises(ValueError):
                preprocess_input_text(bad_input)

        with self.assertRaises(ValueError):
            self.service.predict("Happy", model_type="invalid_model")


class TestAPIFailureTransparency(unittest.TestCase):
    def test_api_rejects_punctuation_and_invalid_model(self):
        with TestClient(backend_main.app) as client:
            r_punct = client.post("/api/v1/predict", json={"text": "!!!...", "model": "bert"})
            self.assertEqual(r_punct.status_code, 400)

            r_bad_model = client.post(
                "/api/v1/predict",
                json={"text": "Good product", "model": "unknown"},
            )
            self.assertEqual(r_bad_model.status_code, 400)

    def test_api_never_falls_back_to_fake_predictions_when_models_missing(self):
        # Simulate completely unavailable ModelService -> must return 503, NEVER fake 88% POSITIVE
        with TestClient(backend_main.app) as client:
            with patch.object(backend_main, "model_service", None):
                h = client.get("/api/v1/health")
                self.assertEqual(h.status_code, 200)
                h_data = h.json()
                self.assertEqual(h_data["status"], "unavailable")
                self.assertFalse(h_data["models_loaded"])

                p = client.post(
                    "/api/v1/predict",
                    json={
                        "text": "The movie is very bad, I don't like it anymore, the characters are not good.",
                        "model": "both",
                    },
                )
                self.assertEqual(p.status_code, 503)

    def test_api_partial_availability_reports_degraded_without_faking_bert(self):
        # Load TF-IDF only while BERT fails (simulates container missing safetensors weights)
        partial_service = ModelService(
            bert_dir=PROJECT_ROOT / "nonexistent_bert_dir",
            tfidf_path=PROJECT_ROOT / "artifacts" / "tfidf" / "tfidf_logistic_regression.joblib",
            lazy=False,
            strict=False,
        )
        with TestClient(backend_main.app) as client:
            with patch.object(backend_main, "model_service", partial_service):
                h = client.get("/api/v1/health").json()
                self.assertEqual(h["status"], "degraded")
                self.assertFalse(h["bert_loaded"])
                self.assertTrue(h["tfidf_loaded"])
                self.assertEqual(h["available_models"], ["tfidf"])

                # Direct bert call -> 503 instead of fake POSITIVE
                r_bert = client.post("/api/v1/predict", json={"text": "Bad service", "model": "bert"})
                self.assertEqual(r_bert.status_code, 503)

                # Direct tfidf call -> genuine prediction
                r_tfidf = client.post(
                    "/api/v1/predict",
                    json={
                        "text": "The movie is very bad, I don't like it anymore, the characters are not good.",
                        "model": "tfidf",
                    },
                )
                self.assertEqual(r_tfidf.status_code, 200)
                self.assertEqual(r_tfidf.json()["sentiment"], "NEGATIVE")

                # Compare both -> real TF-IDF + explicit bert.unavailable flag
                r_both = client.post(
                    "/api/v1/predict",
                    json={
                        "text": "The movie is very bad, I don't like it anymore, the characters are not good.",
                        "model": "both",
                    },
                )
                self.assertEqual(r_both.status_code, 200)
                data = r_both.json()
                self.assertTrue(data["bert"]["unavailable"])
                self.assertEqual(data["tfidf"]["sentiment"], "NEGATIVE")


if __name__ == "__main__":
    unittest.main()
