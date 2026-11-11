import json
from pathlib import Path

from models.email_classifier import evaluate, load_emails


if __name__ == "__main__":
    data = Path(__file__).parent / "data" / "emails.csv"
    print(json.dumps(evaluate(load_emails(data)), indent=2, sort_keys=True))
