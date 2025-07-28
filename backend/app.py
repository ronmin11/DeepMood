from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import io
from PIL import Image
import numpy as np
from together import Together
import getpass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Together AI client
def get_together_client():
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not found in environment variables. Please check your .env file.")
    return Together(api_key=api_key)

# Initialize emotion prediction model
def load_emotion_model():
    try:
        from transformers.pipelines import pipeline
        # You can replace this with your custom model path
        model = pipeline("image-classification", model="google/vit-base-patch16-224")
        return model
    except Exception as e:
        print(f"Error loading emotion model: {e}")
        return None

emotion_model = load_emotion_model()
together_client = get_together_client()

@app.route('/predict', methods=['POST'])
def predict_emotion():
    """Predict emotion from uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Convert image to PIL Image
        image = Image.open(file.stream)
        
        # Predict emotion using the model
        if emotion_model:
            predictions = emotion_model(image)
            # Get the top prediction
            top_prediction = predictions[0] if predictions else None
            
            if top_prediction:
                return jsonify({
                    'emotion': top_prediction['label'],
                    'confidence': top_prediction['score']
                })
            else:
                return jsonify({'error': 'No prediction available'}), 500
        else:
            # Fallback to a mock prediction for testing
            return jsonify({
                'emotion': 'neutral',
                'confidence': 0.85
            })
            
    except Exception as e:
        print(f"Error in emotion prediction: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """Handle chatbot conversations"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        predicted_emotion = data.get('emotion', 'neutral')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Create chat context
        chat = [
            {"role": 'system', 'content': f'You are a helpful therapist, assisting people based on their emotion. The user is {predicted_emotion}.'},
            {"role": 'user', 'content': user_message},
        ]
        
        # Get response from Together AI
        try:
            completion = together_client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=chat,
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9,
            )
            
            # Simple approach: convert to string and extract content
            completion_str = str(completion)
            print(f"Together AI response: {completion_str}")
            
            # For now, use a contextual response based on emotion
            if predicted_emotion.lower() in ['happy', 'joy', 'excited']:
                reply = f"I can sense your positive energy! It's wonderful that you're feeling {predicted_emotion}. What's bringing you this joy today? I'd love to hear more about what's making you feel so good."
            elif predicted_emotion.lower() in ['sad', 'down', 'depressed']:
                reply = f"I notice you might be feeling {predicted_emotion}. It's completely okay to feel this way, and I'm here to listen without judgment. Would you like to talk about what's on your mind?"
            elif predicted_emotion.lower() in ['angry', 'frustrated', 'mad']:
                reply = f"I can see you're feeling {predicted_emotion}. That's a valid emotion, and it's important to acknowledge it. What's been happening that's causing these feelings?"
            elif predicted_emotion.lower() in ['anxious', 'worried', 'nervous']:
                reply = f"I sense some {predicted_emotion} energy from you. Anxiety can be really challenging. Would you like to talk about what's causing these feelings of worry?"
            else:
                reply = f"I understand you're feeling {predicted_emotion}. That's a valid emotion, and I'm here to listen. Can you tell me more about what's on your mind?"
            
        except Exception as e:
            print(f"Error calling Together AI: {e}")
            reply = "I'm sorry, I couldn't generate a response right now. Please try again later."
        
        return jsonify({
            'reply': reply,
            'emotion': predicted_emotion
        })
        
    except Exception as e:
        print(f"Error in chatbot: {e}")
        return jsonify({'error': 'Chatbot failed'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'DeepMood API is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 