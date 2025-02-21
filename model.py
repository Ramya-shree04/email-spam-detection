# Import Libraries
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download("stopwords")

# Load Dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]  # Only keep relevant columns
df.columns = ["label", "message"]

# Convert labels to binary format
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Data Preprocessing
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])  # Remove stopwords
    return text

df["clean_message"] = df["message"].apply(clean_text)

# Convert Text to Vectors (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_message"])
y = df["label"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model (Naïve Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))


def predict_spam(email):
    email = clean_text(email)
    email_vector = vectorizer.transform([email])
    prediction = model.predict(email_vector)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Example Test
test_email = "hi!how are you?"
print(f"Test Email Prediction: {predict_spam(test_email)}")
