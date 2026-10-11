"""
Fast, Memory-Efficient Training Script for Quote Next-Word Prediction
Trains an LSTM model on qoute_dataset.csv and saves both `my_model.keras` and `lstm_model (1).h5`.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import pickle
import string
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "qoute_dataset.csv")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
MODEL_KERAS_PATH = os.path.join(BASE_DIR, "my_model.keras")
MODEL_H5_PATH = os.path.join(BASE_DIR, "lstm_model (1).h5")
MAXLEN_PATH = os.path.join(BASE_DIR, "max_len.pkl")

VOCAB_SIZE = 10_000
EMBEDDING_DIM = 64
RNN_UNITS = 128
MAX_SEQ_LEN = 25   # Sequence context window (90% of quotes are <= 25 words)
EPOCHS = 3
BATCH_SIZE = 256

def main():
    print("=" * 60)
    print("Starting LSTM Model Training for Text Prediction")
    print("=" * 60)

    # 1. Load Tokenizer
    print("\n[1/5] Loading tokenizer...")
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    print(f"  * Tokenizer loaded with {len(tokenizer.word_index):,} words in vocabulary.")

    # 2. Load & Preprocess Dataset
    print("\n[2/5] Loading quotes dataset...")
    df = pd.read_csv(DATASET_PATH)
    quotes = df["quote"].dropna().str.lower()
    translator = str.maketrans("", "", string.punctuation)
    quotes = quotes.apply(lambda x: x.translate(translator))
    print(f"  * Loaded {len(quotes):,} quotes.")

    # 3. Create Training Sequences
    print("\n[3/5] Generating training sequences...")
    t0 = time.time()
    sequences = tokenizer.texts_to_sequences(quotes)

    X, y = [], []
    for seq in sequences:
        for i in range(1, len(seq)):
            sub = seq[:i]
            if len(sub) > MAX_SEQ_LEN:
                sub = sub[-MAX_SEQ_LEN:]
            X.append(sub)
            y.append(seq[i])

    X = pad_sequences(X, maxlen=MAX_SEQ_LEN, padding="pre")
    y = np.array(y, dtype=np.int32)
    print(f"  * Created {len(X):,} sequences in {time.time() - t0:.2f}s.")
    print(f"  * Input shape: {X.shape}, Target shape: {y.shape}")

    # Save max_len (storing MAX_SEQ_LEN + 1 so max_len - 1 == MAX_SEQ_LEN)
    with open(MAXLEN_PATH, "wb") as f:
        pickle.dump(MAX_SEQ_LEN + 1, f)
    print(f"  * Saved max_len={MAX_SEQ_LEN + 1} to {MAXLEN_PATH}")

    # 4. Build Model
    print("\n[4/5] Building LSTM architecture...")
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        LSTM(RNN_UNITS),
        Dense(VOCAB_SIZE, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # 5. Train Model
    print(f"\n[5/5] Training for {EPOCHS} epochs (batch size {BATCH_SIZE})...")
    t_train = time.time()
    model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.05,
        verbose=1
    )
    total_time = time.time() - t_train
    print(f"  * Training completed in {total_time:.1f}s ({total_time/60:.1f} min).")

    # 6. Save Model to both paths so all references work
    print("\nSaving models...")
    try:
        model.save(MODEL_KERAS_PATH)
        print(f"  * Saved {MODEL_KERAS_PATH} ({os.path.getsize(MODEL_KERAS_PATH)/(1024*1024):.2f} MB)")
    except Exception as e:
        print(f"  * Error saving keras model: {e}")

    try:
        model.save(MODEL_H5_PATH)
        print(f"  * Saved {MODEL_H5_PATH} ({os.path.getsize(MODEL_H5_PATH)/(1024*1024):.2f} MB)")
    except Exception as e:
        print(f"  * Note for H5 saving: {e}")

    # 7. Quick Smoke Test
    print("\nRunning verification test...")
    test_phrase = "the only thing we"
    clean_test = test_phrase.translate(translator).lower()
    test_tokens = tokenizer.texts_to_sequences([clean_test])[0]
    test_padded = pad_sequences([test_tokens], maxlen=MAX_SEQ_LEN, padding="pre")
    preds = model.predict(test_padded, verbose=0)[0]
    top3_idx = np.argsort(preds)[-3:][::-1]
    top3 = [(tokenizer.index_word.get(idx, "<unk>"), float(preds[idx])) for idx in top3_idx]
    print(f"  Seed: '{test_phrase}'")
    for word, prob in top3:
        print(f"    -> '{word}' ({prob*100:.1f}%)")

    print("\nAll done! Model files are ready.")

if __name__ == "__main__":
    main()
