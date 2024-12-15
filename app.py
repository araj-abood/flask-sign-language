from flask import Flask, request, jsonify
from neural_hand_detector import NeuralHandSignDetector
from flask_cors import CORS
import numpy as np
import cv2
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize detector and load model
detector = NeuralHandSignDetector()
try:
    detector.load_model('neural_hand_signs.pth')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Warning: No model found! Please train the model first.")

def decode_image(base64_string):
    """Convert base64 image data to OpenCV format"""
    # Remove the data URL prefix if present
    if 'data:image' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64 string to image
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode image
        image = decode_image(data['image'])
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Make prediction
        sign, confidence = detector.predict_sign(image)
        
        return jsonify({
            'sign': sign,
            'confidence': float(confidence)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)