from flask import Flask, render_template, request
import joblib
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load trained model and vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        email_text = request.form["email_text"]
        cleaned_text = clean_text(email_text)
        transformed_text = vectorizer.transform([cleaned_text])
        result = model.predict(transformed_text)
        prediction = "Spam" if result[0] == 1 else "Not Spam"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)  # Render uses dynamic ports
