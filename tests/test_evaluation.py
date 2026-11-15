"""
Evaluation Pipeline & Metrics Tests
Verifies that dataset loader and evaluation functions correctly produce
standardized classification metrics consistent with SST-2 GLUE benchmark.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate import load_dataset, evaluate_tfidf
from backend.app.services.model_loader import load_tfidf_pipeline


class TestEvaluation(unittest.TestCase):
    def test_load_dataset(self):
        dataset_dir = PROJECT_ROOT / "data" / "sst2"
        df = load_dataset(dataset_dir, "validation")
        self.assertEqual(len(df), 872)
        self.assertIn("sentence", df.columns)
        self.assertIn("label", df.columns)
        self.assertEqual(set(df["label"].unique()), {0, 1})

    def test_evaluate_tfidf_validation(self):
        dataset_dir = PROJECT_ROOT / "data" / "sst2"
        df = load_dataset(dataset_dir, "validation")
        texts = df["sentence"].tolist()
        y_true = df["label"].to_numpy()

        tfidf_path = PROJECT_ROOT / "artifacts" / "tfidf" / "tfidf_logistic_regression.joblib"
        pipeline = load_tfidf_pipeline(tfidf_path)

        results = evaluate_tfidf(pipeline, texts, y_true)

        self.assertEqual(results["sample_count"], 872)
        # Verify TF-IDF accuracy is between 80% and 83%
        self.assertTrue(0.80 <= results["accuracy"] <= 0.83)
        self.assertTrue(0.81 <= results["f1_score"] <= 0.84)
        self.assertLess(results["avg_latency_ms"], 50.0)
        self.assertEqual(len(results["confusion_matrix"]), 2)


if __name__ == "__main__":
    unittest.main()
