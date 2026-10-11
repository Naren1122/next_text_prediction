# 🧠 Next Word Prediction (LSTM)

An interactive deep learning application for predicting the next word in a phrase using an **LSTM (Long Short-Term Memory)** neural network trained on famous quotes. Built with **TensorFlow / Keras** and deployed with a modern UI in **Streamlit**.

---

## 📌 Features

- **Real-Time Word Prediction**: Type any prompt and get instantaneous next-word completions.
- **Top Candidates & Probability Distribution**: Displays the highest likelihood candidates and their prediction confidence.
- **Multi-Word Auto-Complete**: Capable of generating full multi-word continuations iteratively.
- **Modern UI**: Custom dark glassmorphic design with intuitive controls.

---

## 📂 Project Structure

```text
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── textpredict.ipynb      # Exploration and prototyping notebook
├── qoute_dataset.csv      # Quotes training dataset
├── my_model.keras         # Trained LSTM model (Keras format)
├── tokenizer.pkl          # Pickled Keras Tokenizer
├── max_len.pkl            # Stored sequence length metadata
├── requirements.txt       # Project dependencies
├── .gitignore             # Excluded cache and temp files
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Naren1122/next_text_prediction.git
cd next_text_prediction
```

### 2. Install Dependencies
Ensure you have Python 3.9+ installed, then install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
Launch the web interface locally:
```bash
streamlit run app.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 Model Training

To retrain the LSTM model on `qoute_dataset.csv`:
```bash
python train_model.py
```
This script preprocesses the dataset, updates `tokenizer.pkl` and `max_len.pkl`, trains an LSTM model, and saves `my_model.keras`.

---

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow, Keras (Embedding, LSTM, Dense)
- **Data Processing**: Pandas, NumPy
- **Frontend / App**: Streamlit
