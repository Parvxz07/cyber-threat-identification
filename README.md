# Cyber Threat Identification System

A Django-based web application that uses NLP and machine learning to classify text (e.g. messages, logs, emails) as a potential **cyber threat** or **benign**, and flags the type of risk detected (phishing, malware reference, social engineering, etc.).

## Features
- Text preprocessing pipeline (tokenization, stopword removal, TF-IDF vectorization)
- Machine learning classifier (Naive Bayes / Logistic Regression) trained on labeled threat/benign text samples
- Simple web UI to paste text and get an instant classification with a confidence score
- Backend logic, model inference, and result rendering handled end-to-end in Django

## Tech Stack
- **Backend:** Python, Django
- **ML/NLP:** scikit-learn, NLTK
- **Frontend:** HTML, CSS
- **Data handling:** pandas

## How it works
1. User submits a block of text through the web form.
2. Text is cleaned and vectorized using TF-IDF.
3. A trained classifier predicts whether the text is a potential security threat.
4. The app displays the prediction, confidence score, and flagged keywords.

## Setup

```bash
git clone https://github.com/Parvxz07/cyber-threat-identification.git
cd cyber-threat-identification
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
python train_model.py      # trains and saves the classifier
python manage.py migrate
python manage.py runserver
```

Visit `http://127.0.0.1:8000/` and paste any text to classify it.

## Project Structure
```
cyber-threat-identification/
├── classifier/
│   ├── model.py          # ML model training & inference
│   ├── preprocess.py     # text cleaning utilities
├── threatapp/
│   ├── views.py
│   ├── urls.py
├── templates/
│   └── index.html
├── static/css/
│   └── style.css
├── data/
│   └── sample_dataset.csv
├── train_model.py
├── manage.py
└── requirements.txt
```

## Notes
This is an educational project built to explore applying NLP/ML techniques to a security-relevant classification problem. It is not a production-grade threat detection system.
