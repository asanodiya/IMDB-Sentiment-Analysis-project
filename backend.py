import re
import pickle
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class Backend:
    def __init__(self):
        self.vocab_size = 10000
        self.max_sequence_length = 100
        self.embedding_dim = 128

        # ✅ tokenizer with vocab limit
        self.tokenizer = Tokenizer(num_words=self.vocab_size)

        self.model = Sequential()

        self.file_path = r'data/IMDB-Train (1).csv'

    # -------------------------
    # Text Cleaning Function
    # -------------------------
    def clean_text(self, review):
        review = review.lower()
        review = re.sub(r'<.*?>', ' ', review)
        review = re.sub(r'[^a-zA-Z]', ' ', review)
        review = re.sub(r'\s+', ' ', review)
        return review.strip()

    # -------------------------
    # Load Data
    # -------------------------
    def load_data(self, file_path):
        data = pd.read_csv(file_path)

        # ✅ apply cleaning
        texts = data['review'].apply(self.clean_text).values

        # ✅ label encoding
        labels = data['sentiment'].map({'positive': 1, 'negative': 0}).values

        return texts, labels

    # -------------------------
    # Preprocess Data
    # -------------------------
    def preprocess_data(self, texts, labels):
        # Tokenization
        self.tokenizer.fit_on_texts(texts)

        sequences = self.tokenizer.texts_to_sequences(texts)

        # Padding
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length
        )

        # ✅ update vocab size dynamically (better)
        self.vocab_size = min(self.vocab_size, len(self.tokenizer.word_index) + 1)

        print("Vocabulary size:", self.vocab_size)

        return padded_sequences, np.array(labels)

    # -------------------------
    # Build Model
    # -------------------------
    def build_model(self):
        self.model.add(
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length
            )
        )
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(64))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(),
            metrics=['accuracy']
        )

    # -------------------------
    # Train Model
    # -------------------------
    def train_model(self, X_train, y_train, X_val, y_val):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=5,
            batch_size=32,
            callbacks=[early_stopping]
        )

    # -------------------------
    # Save Model
    # -------------------------
    def save_model(self, model_path='saved_model/model.h5'):
        self.model.save(model_path)

    def save_tokenizer(self, tokenizer_path='saved_model/tokenizer.pkl'):
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

    # -------------------------
    # Predict
    # -------------------------
    def predict(self, text):
        # ✅ apply same cleaning
        text = self.clean_text(text)

        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length)

        prediction = self.model.predict(padded_sequence)

        if prediction > 0.5:
            return "Positive "
        else:
            return "Negative "


print("✅ Backend class ready with preprocessing!")