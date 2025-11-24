import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb


# ---------------------------
# PARAMETERS (must match training notebook)
# ---------------------------
VOCAB_SIZE = 10000
MAXLEN = 200
INDEX_OFFSET = 3


# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("imdb_lstm_rnn.h5")

model = load_model()


# ---------------------------
# LOAD WORD INDEX
# ---------------------------
@st.cache_resource
def load_word_index():
    return imdb.get_word_index()

word_index = load_word_index()


# ---------------------------
# ENCODE FUNCTION
# ---------------------------
def encode_review(text):
    words = text.lower().split()
    encoded = [1]  # <START>

    for word in words:
        if word in word_index:
            idx = word_index[word] + INDEX_OFFSET
            if idx < VOCAB_SIZE:
                encoded.append(idx)
            else:
                encoded.append(2)  # <UNK>
        else:
            encoded.append(2)  # <UNK>

    padded = pad_sequences([encoded], maxlen=MAXLEN)
    return padded


# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def predict_review(text):
    seq = encode_review(text)
    prob = model.predict(seq, verbose=0)[0][0]
    label = "Positive" if prob >= 0.5 else "Negative"
    return label, float(prob)


# ---------------------------
# STREAMLIT UI SETTINGS
# ---------------------------
st.set_page_config(
    page_title="IMDB Sentiment Classifier",
    page_icon="🎬",
    layout="centered"
)

# Inject Custom CSS
st.markdown("""
<style>
    .main { background-color: #1E1E1E; color: white; }
    textarea { border-radius: 10px !important; }
    div.stButton > button {
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        background-color: #6C63FF;
        color: white;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #5548e0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------
# TITLE
# ---------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:white;'>
        🎬 IMDB Movie Review Classifier <br>
        by <span style='color:#6C63FF;'>Sufiyanov</span>
    </h1>
    <p style='text-align:center; font-size:18px; color: #BBBBBB;'>
        This app uses an LSTM-based RNN model trained on the IMDB dataset to classify movie reviews.
    </p>
    """,
    unsafe_allow_html=True
)


# ---------------------------
# INPUT SECTION
# ---------------------------
st.markdown("### 📝 Enter Your Movie Review")

user_input = st.text_area(
    "Type a movie review below:",
    height=150,
    placeholder="Example: I really loved this movie! The story was emotional and the acting was amazing."
)

if st.button("🔍 Classify Review"):
    if len(user_input.strip()) == 0:
        st.warning("Please write a review to classify.")
    else:
        label, prob = predict_review(user_input)
        color = "#00C851" if "Positive" in label else "#ff4444"

        st.markdown(
            f"""
            <div style="
                padding: 15px;
                border-radius: 10px;
                background-color:{color}33;
                border: 1px solid {color};
                margin-top:15px;">
                <h3 style="color:{color};">Prediction: {label}</h3>
                <p style="color:white;">Confidence: {prob:.4f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------------------
# EXAMPLE REVIEWS
# ---------------------------
st.markdown("## ⭐ Example Reviews and Predictions")

sample_reviews = [
    "This movie was amazing! The storyline was strong and the acting was great.",
    "Absolutely terrible. Waste of time, the plot was horrible.",
    "Not bad, but not great. It was okay overall.",
    "What a beautiful movie. The visuals and music were stunning.",
    "I hated this film. It was boring and predictable."
]

for i, text in enumerate(sample_reviews, start=1):
    label, prob = predict_review(text)
    pred_color = "#00C851" if "Positive" in label else "#ff4444"

    st.markdown(
        f"""
        <div style="
            background-color:#2C2C2C;
            padding:15px;
            border-radius:10px;
            margin-bottom:10px;">
            <h4 style="color:white;">Review {i}</h4>
            <p style="color:#CCCCCC;">{text}</p>
            <p style="color:{pred_color};"><b>Prediction:</b> {label} — Confidence {prob:.3f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
