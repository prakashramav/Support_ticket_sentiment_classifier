"""
Unit Tests for Model Loader & ModelService
Support-Ticket Sentiment Classifier
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.model_loader import (
    load_bert_model,
    load_tfidf_pipeline,
    ModelService,
)


class TestModelLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ModelService(lazy=False)

    def test_bert_loader_artifacts(self):
        bert_dir = PROJECT_ROOT / "artifacts" / "bert"
        tokenizer, model, device = load_bert_model(bert_dir)
        self.assertIsNotNone(tokenizer)
        self.assertIsNotNone(model)
        self.assertIsNotNone(device)
        self.assertFalse(model.training)

    def test_tfidf_loader_artifacts(self):
        tfidf_path = PROJECT_ROOT / "artifacts" / "tfidf" / "tfidf_logistic_regression.joblib"
        pipeline = load_tfidf_pipeline(tfidf_path)
        self.assertIsNotNone(pipeline)
        self.assertTrue(hasattr(pipeline, "predict"))
        self.assertTrue(hasattr(pipeline, "predict_proba"))

    def test_predict_bert_positive(self):
        text = "The customer support team resolved my issue immediately. Very satisfied!"
        res = self.service.predict_bert(text)

        self.assertEqual(res["text"], text)
        self.assertIn("sentiment", res)
        self.assertEqual(res["sentiment"], "POSITIVE")
        self.assertTrue(0.0 <= res["confidence"] <= 1.0)
        self.assertTrue(0 <= res["confidencePercentage"] <= 100)
        self.assertIn("probabilities", res)
        total_prob = res["probabilities"]["POSITIVE"] + res["probabilities"]["NEGATIVE"]
        self.assertAlmostEqual(total_prob, 1.0, places=3)
        self.assertGreater(res["latencyMs"], 0)
        self.assertGreater(len(res["tokens"]), 0)

    def test_predict_tfidf_negative(self):
        text = "The service was completely unacceptable and the product broke on day one."
        res = self.service.predict_tfidf(text)

        self.assertIn("sentiment", res)
        self.assertEqual(res["sentiment"], "NEGATIVE")
        self.assertTrue(0.0 <= res["confidence"] <= 1.0)
        self.assertTrue(0 <= res["confidencePercentage"] <= 100)
        self.assertIn("probabilities", res)
        total_prob = res["probabilities"]["POSITIVE"] + res["probabilities"]["NEGATIVE"]
        self.assertAlmostEqual(total_prob, 1.0, places=3)
        self.assertGreater(res["latencyMs"], 0)

    def test_predict_both(self):
        text = "I am very pleased with the prompt response."
        res = self.service.predict(text, model_type="both")

        self.assertIn("bert", res)
        self.assertIn("tfidf", res)
        self.assertEqual(res["bert"]["sentiment"], "POSITIVE")
        self.assertEqual(res["tfidf"]["sentiment"], "POSITIVE")

    def test_predict_empty_text(self):
        with self.assertRaises(ValueError):
            self.service.predict_bert("   ")

        with self.assertRaises(ValueError):
            self.service.predict_tfidf("")


if __name__ == "__main__":
    unittest.main()
