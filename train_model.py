import pandas as pd
import re
import nltk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv("email_data.csv")  # Ensure this file exists

# Phishing-related keywords with weights
spam_keywords = {
    "urgent": 5, "compromised": 5, "verify": 4, "click": 3, "secure": 3,
    "free": 2, "win": 2, "account": 3, "suspended": 4
}

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Count phishing words
def count_spam_words(text):
    words = text.split()
    return sum(spam_keywords.get(word, 0) for word in words)

# Apply preprocessing
df["Message"] = df["Message"].apply(clean_text)
df["spam_word_count"] = df["Message"].apply(count_spam_words)
df["Category"] = df["Category"].map({"ham": 0, "spam": 1})  # Convert labels to 0 (ham) and 1 (spam)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df[["Message", "spam_word_count"]], df["Category"], test_size=0.2, random_state=42)

# Transform text with TF-IDF
vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"))
X_train_tfidf = vectorizer.fit_transform(X_train["Message"])

# Combine TF-IDF features with spam word count
import scipy.sparse
X_train_combined = scipy.sparse.hstack((X_train_tfidf, X_train["spam_word_count"].values.reshape(-1,1)))

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_combined, y_train)

# Save model and vectorizer
joblib.dump(model, "spam_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model training complete! Saved as 'spam_classifier.pkl'.")
