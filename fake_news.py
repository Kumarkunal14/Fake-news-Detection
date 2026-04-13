import pandas as pd
import numpy as np

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Dataset
# Download dataset from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
real["label"] = 1

# Combine datasets
data = pd.concat([fake, real])

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Use only text column
X = data["text"]
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test with custom input
def predict_news(news):
    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)
    
    if prediction[0] == 0:
        return "Fake News ❌"
    else:
        return "Real News ✅"

# Example
print(predict_news("India launches new satellite for communication"))