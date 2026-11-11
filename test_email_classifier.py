import json
import tempfile
import unittest
from pathlib import Path

from app import create_app
from models.email_classifier import evaluate, load_emails


class EmailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_page_and_ordinary_classification(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        reply = self.client.post("/api/classify", json={"email_text": "Team meeting tomorrow"})
        self.assertEqual(reply.status_code, 200)
        result = reply.get_json()
        self.assertIn(result["category"], ("Work", "Personal", "Spam"))
        self.assertIn(result["priority"], ("High", "Medium", "Low"))
        self.assertTrue(0 <= result["confidence"] <= 1)
        self.assertNotIn("email_text", result)

    def test_invalid_input_is_rejected(self):
        for payload in (None, [], {}, {"email_text": "   "}, {"email_text": 123}):
            with self.subTest(payload=payload):
                reply = self.client.post("/api/classify", data=json.dumps(payload),
                                         content_type="application/json")
                self.assertEqual(reply.status_code, 400)
                self.assertIn("error", reply.get_json())
        reply = self.client.post("/api/classify", json={"email_text": "x" * 10001})
        self.assertEqual(reply.status_code, 400)

    def test_unknown_words_and_large_requests(self):
        reply = self.client.post("/api/classify", json={"email_text": "zzzzxxqqq"})
        self.assertEqual(reply.status_code, 422)
        reply = self.client.post("/api/classify", data="x" * (65 * 1024),
                                 content_type="application/json")
        self.assertEqual(reply.status_code, 413)

    def test_metrics_are_repeatable_and_cover_all_classes(self):
        frame = load_emails(Path(__file__).parent / "data" / "emails.csv")
        first = evaluate(frame)
        self.assertEqual(first, evaluate(frame))
        for report in first["evaluation"].values():
            self.assertEqual(report["train_examples"], 24)
            self.assertEqual(report["test_examples"], 6)
            self.assertEqual(len(report["test_class_counts"]), 3)

    def test_malformed_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / "invalid.csv"
            data.write_text("email_text,category,priority\nhello,Work,\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_emails(data)
