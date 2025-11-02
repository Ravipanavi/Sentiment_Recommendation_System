import joblib
import os
import numpy as np
import pandas as pd

# === Model Loader ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models")

# Load sentiment model and TF-IDF vectorizer
sentiment_model = joblib.load(os.path.join(MODEL_PATH, "sentiment_model.pkl"))
tfidf = joblib.load(os.path.join(MODEL_PATH, "tfidf.pkl"))

# Load recommender dictionary (if available)
try:
    recommender_dict = joblib.load(os.path.join(MODEL_PATH, "recommender.pkl"))
except FileNotFoundError:
    recommender_dict = {}  # fallback if recommender file not found

# === Prediction Functions ===
def predict_sentiment(review_text):
    """Predict sentiment (Positive/Negative) for input review text."""
    review_vec = tfidf.transform([review_text])
    pred = sentiment_model.predict(review_vec)[0]
    sentiment = "Positive" if pred == 1 else "Negative"
    return sentiment

def recommend_products(username):
    """Return top 5 recommended products for a given username."""
    if username in recommender_dict:
        return recommender_dict[username][:5]
    else:
        return ["No recommendations available for this user"]
