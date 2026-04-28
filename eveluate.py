from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import re

# -------------------------
# Load model & tokenizer
# -------------------------
model = load_model('saved_model/model.h5')

with open('saved_model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# -------------------------
# Same preprocessing (VERY IMPORTANT)
# -------------------------
def clean_text(review):
    review = review.lower()
    review = re.sub(r'<.*?>', ' ', review)
    review = re.sub(r'[^a-zA-Z]', ' ', review)
    review = re.sub(r'\s+', ' ', review)
    return review.strip()

# -------------------------
# Load data
# -------------------------
data = pd.read_csv(r'data/IMDB-Test (1).csv')

texts = data['review'].apply(clean_text)
labels = data['sentiment'].map({'positive':1, 'negative':0}).values

# -------------------------
# Convert text → sequences
# -------------------------
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=100)

y_true = labels

# -------------------------
# Predictions
# -------------------------
y_pred_prob = model.predict(X)

# convert probability → class (0/1)
y_pred = (y_pred_prob > 0.5).astype(int)

# -------------------------
# Metrics
# -------------------------
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("✅ Accuracy:", accuracy)
print("✅ F1 Score:", f1)