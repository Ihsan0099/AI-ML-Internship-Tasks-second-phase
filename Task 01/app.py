# app.py
from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Load your model and vectorizer
try:
    model = joblib.load('logistic_regression_model2.pkl')
    vectorizer = joblib.load('vectorizer (1).pkl')
except Exception as e:
    print("❌ Error loading model or vectorizer:", e)
    exit(1)

# Label mapping
label_mapping = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

# Routes
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    title = data.get('title', '').strip()

    if not title:
        return jsonify({"success": False, "error": "No title provided."})

    try:
        # Transform input
        features = vectorizer.transform([title]).toarray()

        # Predict
        prediction = model.predict(features)[0]
        label = label_mapping[prediction]

        # Confidence
        confidence = None
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(features)[0]
            confidence = round(float(np.max(prob)), 2)

        return jsonify({
            "success": True,
            "label": label,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# Optional: Serve static files if needed
@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    print("🚀 Flask app running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)