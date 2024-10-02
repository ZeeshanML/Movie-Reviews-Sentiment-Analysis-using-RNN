import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

model = load_model('simple_rnn_imdb.keras')

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(review):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_review = preprocess_text(review)
    prediction = model.predict(preprocessed_review)[0][0]
    sentiment = "positive" if prediction > 0.5 else "negative"
    return sentiment, prediction

st.title("IMDb Review Sentiment Analysis")
review = st.text_area("Enter your review:")

if st.button("Predict"):
    sentiment, confidence = predict_sentiment(review)
    st.write(f"The review is {sentiment} with confidence {confidence:.2f}")



