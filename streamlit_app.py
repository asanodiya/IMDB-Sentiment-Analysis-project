import streamlit as st
import requests

st.title("🎬 IMDB Sentiment Analysis")

user_input = st.text_area("Enter movie review:")

if st.button("Predict"):
    if user_input.strip() != "":
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"text": user_input}
        )

        if response.status_code == 200:
            result = response.json()

            st.success(f"Sentiment: {result['sentiment']}")
            st.write(f"Confidence: {result['confidence']:.4f}")
        else:
            st.error("API Error")
    else:
        st.warning("Please enter text")