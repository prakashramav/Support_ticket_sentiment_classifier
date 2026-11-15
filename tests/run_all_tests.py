"""
Unified Test Runner for Support-Ticket Sentiment Classifier
Executes all unit, integration, and evaluation tests.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_evaluation import TestEvaluation
from tests.test_api import TestAPIEndpoints
from tests.test_model_loader import TestModelLoader


def run_tests():
    print("=" * 70)
    print("SUPPORT-TICKET SENTIMENT CLASSIFIER - AUTOMATED TEST SUITE")
    print("=" * 70)

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    # Load test cases in logical dependency order
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluation))
    suite.addTests(loader.loadTestsFromTestCase(TestModelLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIEndpoints))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print(f"TEST RUN SUMMARY: {result.testsRun} tests executed.")
    if result.wasSuccessful():
        print("[SUCCESS] All tests passed with 0 errors and 0 failures!")
        print("=" * 70)
        return 0
    else:
        print(f"[FAILURE] Errors: {len(result.errors)}, Failures: {len(result.failures)}")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
