from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and vectorizer
model = joblib.load('chatbot_model.pkl')
vectorizer = joblib.load('chatbot_vectorizer.pkl')

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    user_input = request.json.get('message')  # Get user input from request
    if user_input:
        # Transform user input to the same format as training data
        user_input_vectorized = vectorizer.transform([user_input])
        # Get model prediction
        response = model.predict(user_input_vectorized)
        return jsonify({"response": response[0]})
    else:
        return jsonify({"response": "Sorry, I didn't understand that."})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
