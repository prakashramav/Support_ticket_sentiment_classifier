"""
API Integration Tests for Support-Ticket Sentiment Classifier
FastAPI backend route tests using starlette TestClient.
"""

import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app


class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("message", data)
        self.assertIn("endpoints", data)

    def test_health_endpoint(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["version"], "1.0.0")
        self.assertIn("device", data)
        self.assertTrue(data["models_loaded"])
        self.assertTrue(data["bert_loaded"])
        self.assertTrue(data["tfidf_loaded"])
        self.assertIn("available_models", data)
        self.assertEqual(data["available_models"], ["bert", "tfidf"])

    def test_predict_bert_positive(self):
        payload = {
            "text": "The agent was incredibly helpful and solved my billing problem right away.",
            "model": "bert",
        }
        response = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["text"], payload["text"])
        self.assertEqual(data["sentiment"], "POSITIVE")
        self.assertGreater(data["confidence"], 0.8)
        self.assertEqual(data["modelKey"], "bert")
        self.assertIn("BERT", data["model"])
        self.assertIn("probabilities", data)
        self.assertGreater(data["probabilities"]["POSITIVE"], 0.8)
        self.assertGreater(data["latencyMs"], 0)

    def test_predict_bert_negative(self):
        payload = {
            "text": "My service has been down for three days and nobody is responding. Worst support ever!",
            "model": "bert",
        }
        response = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["sentiment"], "NEGATIVE")
        self.assertGreater(data["confidence"], 0.8)
        self.assertEqual(data["modelKey"], "bert")

    def test_predict_tfidf(self):
        payload = {
            "text": "The delivery was delayed and the package arrived damaged.",
            "model": "tfidf",
        }
        response = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["sentiment"], "NEGATIVE")
        self.assertEqual(data["modelKey"], "tfidf")
        self.assertIn("TF-IDF", data["model"])

    def test_predict_both_models(self):
        payload = {
            "text": "Thank you for the wonderful customer experience!",
            "model": "both",
        }
        response = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("bert", data)
        self.assertIn("tfidf", data)
        self.assertEqual(data["bert"]["sentiment"], "POSITIVE")
        self.assertEqual(data["tfidf"]["sentiment"], "POSITIVE")

    def test_predict_empty_text_error(self):
        payload = {
            "text": "   ",
            "model": "bert",
        }
        response = self.client.post("/api/v1/predict", json=payload)
        self.assertIn(response.status_code, [400, 422])

    def test_predict_missing_text_error(self):
        payload = {
            "model": "bert",
        }
        response = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
