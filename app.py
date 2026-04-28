from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

app = FastAPI()

# -------------------------
# Load model & tokenizer
# -------------------------
model = load_model("saved_model/model.h5")

with open("saved_model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_sequence_length = 100

# -------------------------
# Cleaning function (same as backend)
# -------------------------
def clean_text(review):
    review = review.lower()
    review = re.sub(r'<.*?>', ' ', review)
    review = re.sub(r'[^a-zA-Z]', ' ', review)
    review = re.sub(r'\s+', ' ', review)
    return review.strip()

# -------------------------
# Request schema
# -------------------------
class ReviewRequest(BaseModel):
    text: str

# -------------------------
# API Endpoint
# -------------------------
@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    text = clean_text(request.text)

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_sequence_length)

    prediction = model.predict(padded)[0][0]

    sentiment = "Positive" if prediction > 0.5 else "Negative"

    return {
        "sentiment": sentiment,
        "confidence": float(prediction)
    }