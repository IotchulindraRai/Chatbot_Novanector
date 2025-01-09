from flask import Flask, request, jsonify, render_template
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Function to load and check the model
def load_model(model_path):
    try:
        # Check if the model file exists before trying to load it
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found!")

        # Try to load the model using joblib
        model = joblib.load(model_path)
        print("Model loaded successfully!")
        return model  # Return the loaded model

    except FileNotFoundError as e:
        # Handle file not found error
        print(f"Error: {str(e)}")
        return None
    except Exception as e:
        # Handle other exceptions such as incompatible model
        print(f"Error while loading the model: {str(e)}")
        return None


# Load the machine learning model (replace with your actual model path)
model_path = "words.pkl"  # Ensure the model.pkl file exists in the same directory or provide the full path
model = load_model(model_path)

# Route to serve the main HTML page
@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is in the "templates" folder

# API route to handle predictions
@app.route('/chatbot', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model is not loaded correctly"}), 500  # Handle model loading failure

    try:
        # Get JSON data from the request
        data = request.json
        query = data.get('query')

        # Check if query is provided
        if not query:
            return jsonify({"error": "Query parameter is missing"}), 400

        # Perform prediction
        response = model.predict([query])  # Assuming your model works with a single query as input
        return jsonify({"response": response[0]})  # Send the response
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
