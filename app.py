import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

# Load model and TF-IDF vectorizer
model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Streamlit UI
st.title("Fake News Detector 🕵️")
st.write("Enter a news article to check if it's real or fake.")

# Input text box
user_input = st.text_area("Paste the news article here:", height=200)

if st.button("Check"):
    if user_input:
        # Preprocess and predict
        cleaned_text = preprocess_text(user_input)
        vectorized_text = tfidf.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        
        # Display result
        st.subheader("Result")
        if prediction == 1:
            st.success("✅ This news is **REAL**.")
        else:
            st.error("❌ This news is **FAKE**.")
        
        # Show confidence score (if model supports predict_proba)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vectorized_text)[0]
            st.write(f"Confidence: {max(proba)*100:.2f}%")
    else:
        st.warning("Please enter a news article!")