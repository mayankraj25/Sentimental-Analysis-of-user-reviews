from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Import CORS
import joblib
import re

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    return text

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data.get("review", "")
    cleaned = clean_text(review)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)
    return jsonify({"sentiment": pred[0]})

if __name__ == "__main__":
    app.run(debug=True)