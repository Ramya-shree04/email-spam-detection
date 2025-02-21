from flask import Flask, render_template, request
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Download stopwords
nltk.download("stopwords")

# Load Dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "message"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Preprocess Text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])
    return text

df["clean_message"] = df["message"].apply(clean_text)

# Train Model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_message"])
y = df["label"]

model = MultinomialNB()
model.fit(X, y)

# Save Model and Vectorizer
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Initialize Flask App
app = Flask(__name__)

# Home Route (Form to Paste Email)
@app.route('/')
def home():
    return render_template("index.html")

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    cleaned_text = clean_text(email_text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    return render_template("index.html", email_text=email_text, prediction=result)

# Run App
if __name__ == '__main__':
    app.run(debug=True)
