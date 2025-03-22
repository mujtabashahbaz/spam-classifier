from flask import Flask, render_template, request
import joblib
import re
import nltk
import scipy.sparse

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load trained model and vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Spam-related keywords
spam_keywords = ["urgent", "compromised", "verify", "click", "secure", "free", "win", "account", "suspended"]

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def count_spam_words(text):
    words = text.split()
    return sum(1 for word in words if word in spam_keywords)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        email_text = request.form["email_text"]
        cleaned_text = clean_text(email_text)
        
        # Transform input
        transformed_text = vectorizer.transform([cleaned_text])
        spam_word_count = count_spam_words(cleaned_text)
        input_features = scipy.sparse.hstack((transformed_text, [[spam_word_count]]))

        # Predict spam or ham
        result = model.predict(input_features)
        prediction = "Spam" if result[0] == 1 else "Not Spam"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)  # Render-compatible port
