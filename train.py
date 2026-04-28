from backend import Backend
from sklearn.model_selection import train_test_split
import os

# create folder if not exists
os.makedirs('saved_model', exist_ok=True)

# initialize
backend = Backend()

# load data
texts, labels = backend.load_data(backend.file_path)

# preprocess
X, y = backend.preprocess_data(texts, labels)

# split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# build model
backend.build_model()

# train
backend.train_model(X_train, y_train, X_val, y_val)

# save model + tokenizer
backend.save_model()
backend.save_tokenizer()

print("✅ Training complete & model saved!")