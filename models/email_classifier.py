from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline


LABELS = {
    "category": {"Work", "Personal", "Spam"},
    "priority": {"High", "Medium", "Low"},
}


def load_emails(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = ["email_text", "category", "priority"]
    if list(frame.columns) != required:
        raise ValueError("Dataset must contain email_text,category,priority.")
    for column in required:
        frame[column] = frame[column].str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"Dataset contains a missing {column}.")
    if frame["email_text"].str.casefold().duplicated().any():
        raise ValueError("Dataset contains repeated email text.")
    for target, labels in LABELS.items():
        if set(frame[target]) != labels or frame[target].value_counts().min() < 2:
            raise ValueError(f"Expected all three {target} classes with at least two examples each.")
    return frame


def new_pipeline(target: str):
    if target == "category":
        return make_pipeline(
            TfidfVectorizer(max_features=500, stop_words="english"),
            MultinomialNB(),
        )
    return make_pipeline(
        TfidfVectorizer(max_features=200, stop_words="english"),
        KNeighborsClassifier(n_neighbors=3),
    )


class EmailModels:
    def __init__(self):
        self.models = {target: new_pipeline(target) for target in LABELS}
        self.trained = False

    def train(self, frame):
        for target, model in self.models.items():
            model.fit(frame["email_text"], frame[target])
        self.trained = True

    def predict(self, text):
        if not self.trained:
            raise ValueError("Models are not trained.")
        for model in self.models.values():
            if model[0].transform([text]).nnz == 0:
                raise ValueError("No recognized words. Try a more descriptive email.")
        category = self.models["category"]
        return {
            "category": str(category.predict([text])[0]),
            "confidence": float(category.predict_proba([text])[0].max()),
            "priority": str(self.models["priority"].predict([text])[0]),
        }


def evaluate(frame):
    results = {}
    for target in LABELS:
        train, test = train_test_split(
            frame, test_size=0.2, random_state=42, stratify=frame[target]
        )
        model = new_pipeline(target)
        model.fit(train["email_text"], train[target])
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit([[0]] * len(train), train[target])
        predictions = model.predict(test["email_text"])
        baseline_predictions = baseline.predict([[0]] * len(test))
        results[target] = {
            "train_examples": len(train),
            "test_examples": len(test),
            "accuracy": accuracy_score(test[target], predictions),
            "macro_f1": f1_score(test[target], predictions, average="macro", zero_division=0),
            "majority_baseline_accuracy": accuracy_score(test[target], baseline_predictions),
            "test_class_counts": {str(k): int(v) for k, v in test[target].value_counts().items()},
        }
    return {"seed": 42, "examples": len(frame), "evaluation": results}
