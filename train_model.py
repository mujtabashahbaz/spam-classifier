import pandas as pd
import re
import nltk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv("email_data.csv")  # Ensure this file exists

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df["Message"] = df["Message"].apply(clean_text)
df["Category"] = df["Category"].map({"ham": 0, "spam": 1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df["Message"], df["Category"], test_size=0.2, random_state=42)

# Train model
vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"))
X_train_tfidf = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save model and vectorizer
joblib.dump(model, "spam_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model training complete! Model saved as 'spam_classifier.pkl'.")
