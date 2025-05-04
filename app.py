from flask import Flask, request, jsonify
from transformers import pipeline
import os

# Load the emotion classification model from Hugging Face
model_name = "leila-may/multi-emotion"
classifier = pipeline("text-classification", model=model_name)

app = Flask(__name__)

@app.route('/')
def home():
    return "Emotion Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the text from the request body
    data = request.get_json()
    text = data.get("inputs")
    
    if not text:
        return jsonify({"error": "No input text provided"}), 400
    
    # Get prediction from the model
    prediction = classifier(text)
    
    # Return the prediction as a response
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
