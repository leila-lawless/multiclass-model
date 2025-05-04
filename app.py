from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)

# Load the emotion classification pipeline
emotion_classifier = pipeline(
    "text-classification",
    model="leila-may/multi-emotion",
    return_all_scores=True
)

@app.route('/')
def home():
    return jsonify({
        "message": "Emotion Classification API",
        "usage": "POST JSON to /predict with 'text' field",
        "example": {"text": "I'm thrilled about the new discoveries!"}
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Please provide a 'text' field in JSON"}), 400
            
        text = data['text']
        predictions = emotion_classifier(text)[0]
        formatted = {item['label']: float(item['score']) for item in predictions}
        return jsonify(formatted)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
