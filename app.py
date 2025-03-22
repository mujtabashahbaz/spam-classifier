from flask import Flask, render_template, request
import joblib
import re
import nltk
import scipy.sparse

# Load NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load pre-trained model and vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Phishing keywords with weights
spam_keywords = {
    "urgent": 5, "compromised": 5, "verify": 4, "click": 3, "secure": 3,
    "free": 2, "win": 2, "account": 3, "suspended": 4
}

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Count phishing words
def count_spam_words(text):
    words = text.split()
    return sum(spam_keywords.get(word, 0) for word in words)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        email_text = request.form["email_text"]
        cleaned_text = clean_text(email_text)
        
        # Rule-based check for phishing
        if any(word in cleaned_text for word in spam_keywords.keys()):
            prediction = "Spam (Phishing detected)"
        else:
            # Machine learning prediction
            transformed_text = vectorizer.transform([cleaned_text])
            spam_word_count = count_spam_words(cleaned_text)
            input_features = scipy.sparse.hstack((transformed_text, [[spam_word_count]]))
            result = model.predict(input_features)
            prediction = "Spam" if result[0] == 1 else "Not Spam"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)  # Render-compatible port
