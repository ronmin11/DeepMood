from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import io
from PIL import Image
import numpy as np
from together import Together
from dotenv import load_dotenv
import cv2
import tempfile
import traceback

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins=['http://localhost:5173', 'http://localhost:8080', 'http://localhost:3000', 'https://deepmood.onrender.com'])

# Initialize Together AI client
def get_together_client():
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("ERROR: TOGETHER_API_KEY not found in environment variables. Please check your .env file.")
        return None
    return Together(api_key=api_key)

# Initialize emotion prediction model (mock for now)
def load_emotion_model():
    try:
        # For now, we'll use a mock model
        # In production, you would load your trained emotion detection model here
        return None
    except Exception as e:
        print(f"Error loading emotion model: {e}")
        return None

# Mock emotion detection function
def detect_emotion_from_image(image):
    """
    Mock emotion detection function.
    In production, this would use your trained model from model.py
    """
    # List of possible emotions
    emotions = ['happy', 'sad', 'angry', 'surprised', 'neutral', 'fearful', 'disgusted']
    confidences = [0.85, 0.78, 0.92, 0.67, 0.73, 0.81, 0.69]
    
    # For demo purposes, return a random emotion
    import random
    emotion_idx = random.randint(0, len(emotions) - 1)
    
    return {
        'emotion': emotions[emotion_idx],
        'confidence': confidences[emotion_idx]
    }

emotion_model = load_emotion_model()
together_client = None

# Initialize Together client with error handling
try:
    together_client = get_together_client()
    if together_client:
        print("Together AI client initialized successfully")
    else:
        print("WARNING: Together AI client not initialized - API key missing")
except Exception as e:
    print(f"ERROR: Failed to initialize Together AI client: {e}")
    together_client = None

@app.route('/predict', methods=['POST'])
def predict_emotion():
    """Predict emotion from uploaded image"""
    try:
        print("Received emotion prediction request")
        # Handle both file upload and base64 image data
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            image = Image.open(file.stream)
        elif request.json and 'image' in request.json:
            # Handle base64 encoded image
            image_data = request.json['image']
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Use mock emotion detection for now
        result = detect_emotion_from_image(image)
        print(f"Emotion detection result: {result}")
        
        return jsonify(result)
            
    except Exception as e:
        print(f"Error in emotion prediction: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """Handle chatbot conversations"""
    try:
        print("Received chatbot request")
        data = request.get_json()
        print(f"Request data: {data}")
        
        user_message = data.get('message', '')
        predicted_emotion = data.get('emotion', 'neutral')
        
        if not user_message:
            print("ERROR: No message provided")
            return jsonify({'error': 'No message provided'}), 400
        
        print(f"Processing message: '{user_message}' with emotion: '{predicted_emotion}'")
        
        # Create chat context with emotion-aware system prompt
        chat = [
            {"role": 'system', 'content': f'You are a compassionate and empathetic AI therapist. The user is currently feeling {predicted_emotion}. Provide supportive, understanding, and helpful responses that acknowledge their emotional state. Be warm, non-judgmental, and offer practical guidance when appropriate. Keep responses conversational and not overly clinical.'},
            {"role": 'user', 'content': user_message},
        ]
        
        # Get response from Together AI
        if together_client:
            print("Using Together AI for response")
            try:
                completion = together_client.chat.completions.create(
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                    messages=chat,
                    max_tokens=1000,
                    temperature=0.7,
                    top_p=0.9,
                )
                
                # Extract the response content
                if completion.choices and len(completion.choices) > 0:
                    reply = completion.choices[0].message.content
                    print(f"Together AI response: {reply[:100]}...")
                else:
                    # Fallback response based on emotion
                    reply = get_fallback_response(predicted_emotion, user_message)
                    print("Using fallback response - no choices in completion")
                
            except Exception as e:
                print(f"Error calling Together AI: {e}")
                traceback.print_exc()
                reply = get_fallback_response(predicted_emotion, user_message)
        else:
            print("Using fallback response - Together AI client not available")
            reply = get_fallback_response(predicted_emotion, user_message)
        
        response_data = {
            'reply': reply,
            'emotion': predicted_emotion
        }
        print(f"Sending response: {response_data}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in chatbot: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Chatbot failed: {str(e)}'}), 500

def get_fallback_response(emotion, message):
    """Generate fallback responses based on emotion when AI service is unavailable"""
    emotion = emotion.lower()
    
    if emotion in ['happy', 'joy', 'excited']:
        return f"I can sense your positive energy! It's wonderful that you're feeling {emotion}. What's bringing you this joy today? I'd love to hear more about what's making you feel so good."
    elif emotion in ['sad', 'down', 'depressed']:
        return f"I notice you might be feeling {emotion}. It's completely okay to feel this way, and I'm here to listen without judgment. Would you like to talk about what's on your mind?"
    elif emotion in ['angry', 'frustrated', 'mad']:
        return f"I can see you're feeling {emotion}. That's a valid emotion, and it's important to acknowledge it. What's been happening that's causing these feelings?"
    elif emotion in ['anxious', 'worried', 'nervous', 'fearful']:
        return f"I sense some {emotion} energy from you. These feelings can be really challenging. Would you like to talk about what's causing these feelings?"
    elif emotion in ['surprised']:
        return f"You seem {emotion}! Sometimes unexpected things can catch us off guard. How are you processing what's happening?"
    else:
        return f"I understand you're feeling {emotion}. That's a valid emotion, and I'm here to listen. Can you tell me more about what's on your mind?"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy', 
            'message': 'DeepMood API is running',
            'together_ai_available': together_client is not None
        })
    except Exception as e:
        print(f"Error in health check: {e}")
        return jsonify({'error': 'Health check failed'}), 500

# Add error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.before_request
def log_request_info():
    print(f"Request: {request.method} {request.url}")
    if request.is_json:
        print(f"Request JSON: {request.get_json()}")

if __name__ == '__main__':
    print("Starting DeepMood Flask application...")
    print(f"Together AI client status: {'Available' if together_client else 'Not available'}")
    app.run(debug=True, host='0.0.0.0', port=5000)