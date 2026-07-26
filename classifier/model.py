"""
Training and inference logic for the cyber threat text classifier.

Pipeline: TF-IDF vectorizer -> Multinomial Naive Bayes classifier.
"""
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from .preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "classifier", "threat_model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "classifier", "vectorizer.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "sample_dataset.csv")


def train():
    """Train the classifier on the sample dataset and persist it to disk."""
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["text"].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Model saved to {MODEL_PATH}")


def load():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict(text: str):
    """Return (label, confidence) for a given piece of text."""
    model, vectorizer = load()
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    label = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = max(proba)
    return label, round(float(confidence) * 100, 2)


if __name__ == "__main__":
    train()
