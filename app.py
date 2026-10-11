import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import pickle
import numpy as np
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Base directory for reliable relative file resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------
# Page Configuration & Aesthetics
# ------------------------------
st.set_page_config(
    page_title="Next Word Prediction (LSTM)",
    page_icon="🧠",
    layout="centered"
)

# Custom Styling (Dark Glassmorphism UI)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f172a 50%, #020617 100%);
    color: #f8fafc;
}

.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #a78bfa 0%, #ec4899 50%, #f43f5e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

.glass-box {
    background: rgba(30, 41, 59, 0.55);
    border: 1px solid rgba(167, 139, 250, 0.2);
    border-radius: 16px;
    padding: 1.75rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}

.result-card {
    background: linear-gradient(135deg, rgba(88, 28, 135, 0.35) 0%, rgba(30, 27, 75, 0.5) 100%);
    border: 1px solid rgba(168, 85, 247, 0.4);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 1.25rem;
}

.result-word {
    font-size: 2.2rem;
    font-weight: 800;
    color: #38bdf8;
    letter-spacing: 0.02em;
    text-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
    margin: 0.5rem 0;
}

.full-sentence {
    color: #cbd5e1;
    font-size: 1.15rem;
    line-height: 1.6;
    margin-top: 0.75rem;
    padding: 0.75rem 1rem;
    background: rgba(15, 23, 42, 0.6);
    border-radius: 8px;
    border-left: 3px solid #a855f7;
}

.candidate-pill {
    display: inline-block;
    background: rgba(147, 51, 234, 0.15);
    border: 1px solid rgba(168, 85, 247, 0.3);
    border-radius: 9999px;
    padding: 0.35rem 0.85rem;
    margin: 0.25rem;
    font-size: 0.88rem;
    color: #e2e8f0;
}

.quick-chip {
    font-size: 0.82rem;
    color: #a5b4fc;
    background: rgba(99, 102, 241, 0.12);
    border: 1px dashed rgba(99, 102, 241, 0.35);
    border-radius: 8px;
    padding: 0.3rem 0.7rem;
    cursor: pointer;
    display: inline-block;
    margin: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load saved files
# ------------------------------
@st.cache_resource(show_spinner="⚡ Loading pre-trained model & resources...")
def load_resources():
    # Attempt to load model from supported names (h5 or keras)
    candidate_paths = [
        os.path.join(BASE_DIR, "lstm_model (1).h5"),
        os.path.join(BASE_DIR, "my_model.keras"),
        os.path.join(BASE_DIR, "lstm_model.h5")
    ]
    
    model = None
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                model = load_model(path)
                break
            except Exception:
                continue

    tok_path = os.path.join(BASE_DIR, "tokenizer.pkl")
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)

    max_len_path = os.path.join(BASE_DIR, "max_len.pkl")
    with open(max_len_path, "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len

model, tokenizer, max_len = load_resources()

# Reverse word lookup for O(1) performance
word_to_index = tokenizer.word_index
index_to_word = tokenizer.index_word

# ------------------------------
# Prediction function
# ------------------------------
def predict_next_word(text):
    if not text or not text.strip():
        return ""
    
    # Preprocess text to match tokenizer vocabulary
    clean_text = text.lower().translate(str.maketrans("", "", string.punctuation)).strip()
    sequences = tokenizer.texts_to_sequences([clean_text])
    
    if not sequences or not sequences[0]:
        return ""

    sequence = sequences[0]
    sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')

    preds = model.predict(sequence, verbose=0)
    predicted_index = int(np.argmax(preds))

    # Fast O(1) lookup with dictionary fallback
    if predicted_index in index_to_word:
        return index_to_word[predicted_index]

    for word, index in word_to_index.items():
        if index == predicted_index:
            return word
    return ""

def predict_top_k(text, k=5):
    """Retrieve top-K candidates with probability distribution."""
    if not text or not text.strip():
        return []
    clean_text = text.lower().translate(str.maketrans("", "", string.punctuation)).strip()
    sequences = tokenizer.texts_to_sequences([clean_text])
    if not sequences or not sequences[0]:
        return []

    sequence = pad_sequences([sequences[0]], maxlen=max_len-1, padding='pre')
    probs = model.predict(sequence, verbose=0)[0]
    top_indices = np.argsort(probs)[-k:][::-1]

    candidates = []
    for idx in top_indices:
        word = index_to_word.get(idx, None)
        if word:
            candidates.append((word, float(probs[idx])))
    return candidates

# ------------------------------
# Streamlit UI
# ------------------------------
st.markdown('<div class="main-title">🧠 Next Word Prediction (LSTM)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a sentence and the model will predict the <b>next word</b> in real time.</div>', unsafe_allow_html=True)

# Sample prompt chips helper
sample_prompts = [
    "The only thing we",
    "To be or not",
    "Life is what happens",
    "In three words I can"
]

if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

# Quick clickable examples
st.markdown("<p style='font-size:0.85rem; color:#94a3b8; margin-bottom:0.4rem;'>💡 Try an example:</p>", unsafe_allow_html=True)
chip_cols = st.columns(len(sample_prompts))
for idx, prompt in enumerate(sample_prompts):
    if chip_cols[idx].button(f"“{prompt}…”", key=f"sample_{idx}", use_container_width=True):
        st.session_state["input_text"] = prompt

user_input = st.text_input(
    "✍️ Enter text:",
    value=st.session_state.get("input_text", ""),
    placeholder="Type a sentence here...",
    key="text_box"
)

col_btn, _ = st.columns([1, 2])
with col_btn:
    predict_clicked = st.button("Predict Next Word", type="primary", use_container_width=True)

if predict_clicked:
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Predicting next word..."):
            next_word = predict_next_word(user_input)
            top_candidates = predict_top_k(user_input, k=5)

        if next_word:
            st.success(f"**Predicted Next Word:** {next_word}")
            
            # Interactive completed sentence card
            completed_phrase = f"{user_input.strip()} **{next_word}**"
            st.markdown(f"""
            <div class="result-card">
                <span style="font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em; color:#c084fc;">
                    Top Candidate
                </span>
                <div class="result-word">{next_word}</div>
                <div class="full-sentence">
                    💬 <em>"{user_input.strip()} <span style='color:#38bdf8; font-weight:700;'>{next_word}</span>"</em>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Optional Top-K Insights
            if top_candidates:
                with st.expander("📊 View Top 5 Predictions & Probabilities"):
                    pills_html = "".join([
                        f"<span class='candidate-pill'><b>{w}</b> ({prob*100:.1f}%)</span>"
                        for w, prob in top_candidates
                    ])
                    st.markdown(f"<div style='margin-top:0.5rem;'>{pills_html}</div>", unsafe_allow_html=True)

                    for word, prob in top_candidates:
                        st.progress(min(prob, 1.0), text=f"**{word}**: {prob*100:.2f}%")
        else:
            st.info("The model could not predict a word for this input. Try using words common in famous quotes!")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("LSTM-based Next Word Prediction using Streamlit")
