"""
Text preprocessing utilities for the cyber threat classifier.
"""
import re
import string


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/punctuation/extra whitespace from input text."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)                  # remove emails
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)                      # remove numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Keywords commonly associated with phishing / social-engineering attempts.
# Used to highlight flagged terms in the UI alongside the model prediction.
SUSPICIOUS_KEYWORDS = [
    "verify", "password", "urgent", "suspended", "click here", "confirm",
    "bank details", "login", "credentials", "prize", "winner", "refund",
    "security alert", "unusual activity", "update immediately", "account locked",
]


def flag_keywords(text: str):
    text_lower = text.lower()
    return [kw for kw in SUSPICIOUS_KEYWORDS if kw in text_lower]
