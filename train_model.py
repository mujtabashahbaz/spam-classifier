import pandas as pd
import re
import nltk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv("email_data.csv")  # Make sure the file exists

# Spam words to check for phishing patterns
spam_keywords = ["urgent", "compromised", "verify", "click", "secure", "free", "win", "account", "suspended"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def count_spam_words(text):
    words = text.split()
    return sum(1 for word in words if word in spam_keywords)

# Apply text cleaning
df["Message"] = df["Message"].apply(clean_text)
df["spam_word_count"] = df["Message"].apply(count_spam_words)
df["Category"] = df["Category"].map({"ham": 0, "spam": 1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df[["Message", "spam_word_count"]], df["Category"], test_size=0.2, random_state=42)

# Transform text using TF-IDF
vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"))
X_train_tfidf = vectorizer.fit_transform(X_train["Message"])

# Combine TF-IDF features with spam word count
import scipy.sparse
X_train_combined = scipy.sparse.hstack((X_train_tfidf, X_train["spam_word_count"].values.reshape(-1,1)))

# Train a better model (Random Forest instead of Naïve Bayes)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_combined, y_train)

# Save the model and vectorizer
joblib.dump(model, "spam_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model training complete! Saved as 'spam_classifier.pkl'.")
