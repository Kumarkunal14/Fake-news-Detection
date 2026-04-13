from flask import Flask, render_template, request
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

app = Flask(__name__)

MODEL_FILE = "model.pkl"

# =========================
# 🧠 Load or Train Model
# =========================
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    print("Training model...")

    # Load dataset (only if available locally)
    fake = pd.read_csv("Fake.csv", low_memory=False)
    real = pd.read_csv("True.csv", low_memory=False)

    fake["label"] = 0
    real["label"] = 1

    # Combine & reduce size
    data = pd.concat([fake, real])
    data = data.sample(n=10000, random_state=42)

    # Create content
    data["content"] = data["title"] + " " + data["text"]

    X = data["content"]
    y = data["label"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Model
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words='english', max_df=0.7)),
        ("clf", PassiveAggressiveClassifier(max_iter=1000))
    ])

    # Train
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, MODEL_FILE)
    print("Model trained and saved!")

# =========================
# 🌐 Flask Routes
# =========================
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    confidence = ""

    if request.method == "POST":
        news = request.form["news"]

        if news.strip() != "":
            result = model.predict([news])
            score = model.decision_function([news])

            conf = abs(score[0])
            conf = min(conf, 1) * 100

            if result[0] == 0:
                prediction = "❌ Fake News"
            else:
                prediction = "✅ Real News"

            confidence = f"Confidence: {conf:.2f}%"
        else:
            prediction = "⚠️ Please enter some text"

    return render_template("index.html", prediction=prediction, confidence=confidence)

# =========================
# ▶️ Run App
# =========================
if __name__ == "__main__":
   app.run(debug=True)

