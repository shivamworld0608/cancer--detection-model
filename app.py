from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
import time
import logging

# Initialize Flask app
app = Flask(__name__)

# Set maximum allowed file size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Path to the trained model
MODEL_PATH = 'trained.keras'

# Initialize logger for debugging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Define a function to check if the uploaded file is valid
def allowed_file(filename):
    if '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        logging.debug(f"File '{filename}' has a valid extension.")
        return True
    logging.warning(f"File '{filename}' has an invalid extension.")
    return False

# Define a preprocessing function to resize and normalize the image
def preprocess_image(image, target_size=(224, 224)):
    try:
        logging.debug(f"Resizing image to {target_size}.")
        image = image.resize(target_size)
        image = np.array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        logging.error(f"Error during image preprocessing: {e}")
        raise ValueError(f"Error during image preprocessing: {e}")

# Health check route for debugging
@app.route('/health', methods=['GET'])
def health():
    logging.debug("Health check request received.")
    return jsonify({'status': 'ok'}), 200

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()

    # Check if a file was uploaded
    if 'file' not in request.files:
        logging.error("No file part in the request.")
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        logging.error("No file selected.")
        return jsonify({'error': 'No selected file'}), 400

    # Validate the file type
    if not allowed_file(file.filename):
        logging.error(f"Invalid file type for file '{file.filename}'.")
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, and JPEG are allowed.'}), 400

    try:
        logging.debug("Reading image from the uploaded file.")
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)

        preprocessing_time = time.time() - start_time
        logging.info(f"Image preprocessing time: {preprocessing_time:.2f} seconds")

        # Predict with the model
        if model is None:
            logging.error("Model is not loaded.")
            return jsonify({'error': 'Model not loaded'}), 500

        prediction_start = time.time()
        predictions = model.predict(processed_image)
        prediction_time = time.time() - prediction_start
        logging.info(f"Model prediction time: {prediction_time:.2f} seconds")

        # Interpret prediction
        prediction_label = 'Cancerous' if predictions[0][0] > 0.5 else 'Non-cancerous'

        # Log total request handling time
        total_time = time.time() - start_time
        logging.info(f"Total request time: {total_time:.2f} seconds")

        # Return the prediction result
        return jsonify({
            'prediction': prediction_label,
            'confidence': float(predictions[0][0]),
            'times': {
                'preprocessing': preprocessing_time,
                'prediction': prediction_time,
                'total': total_time
            }
        })

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    # Use PORT from the environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    logging.info(f"Starting Flask app on port {port}...")
    app.run(host='0.0.0.0', port=port, threaded=True)
