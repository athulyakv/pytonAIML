import streamlit as st
import pickle

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Streamlit UI
st.set_page_config(page_title="Spam Detector", layout="centered")
st.title("📩 Spam Message Detector (Naive Bayes)")

message = st.text_area("Enter your message here:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        msg_vec = vectorizer.transform([message])
        pred = model.predict(msg_vec)[0]
        result = "🚫 Spam" if pred == 1 else "✅ Not Spam"
        st.success(f"Prediction: *{result}*")