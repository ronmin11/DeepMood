from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from together import Together
from dotenv import load_dotenv
import logging

# Import your existing model classes
from model import EmotionNet, get_model_info

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Enable CORS for all routes with comprehensive configuration
CORS(app, 
     origins=['http://localhost:3000', 'http://localhost:8080', 'http://localhost:8081', 'http://127.0.0.1:8080', 'http://127.0.0.1:8081'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model
model = None
device = None
transform = None
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Initialize Together AI client
together_client = None

def initialize_model():
    """Initialize the emotion detection model using your existing structure"""
    global model, device, transform
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Get model info from your existing structure
        model_name, num_classes = get_model_info()
        logger.info(f"Using model: {model_name}")
        
        # Create backbone using your existing approach
        backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        
        # Create EmotionNet using your existing class
        model = EmotionNet(backbone, num_classes)
        
        # Load the trained weights - handle float16 model
        checkpoint_path = 'best_model_float16.pth'
        if os.path.exists(checkpoint_path):
            try:
                # Load checkpoint to the specified device (CPU/GPU)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # Handle state dict from checkpoint
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Clean up state dict keys (remove 'module.' prefix if present from DataParallel)
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                # Load the state dict
                model.load_state_dict(state_dict, strict=False)
                logger.info("Float16 model weights loaded successfully")
                
                # Convert model to float16 for inference
                model = model.half()
                
                # Move model to the appropriate device
                model = model.to(device)
                
            except Exception as e:
                logger.error(f"Error loading float16 model weights: {str(e)}")
                return False
        else:
            logger.error(f"Model file {checkpoint_path} not found.")
            return False
        
        model.eval()
        
        # Create transform compatible with your model
        data_config = timm.data.resolve_model_data_config(backbone)
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(data_config['mean'], data_config['std']),
        ])
        
        logger.info("Model initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return False

def initialize_together_ai():
    """Initialize Together AI client for chatbot"""
    global together_client
    
    try:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            logger.error("TOGETHER_API_KEY not found in environment variables")
            return False
            
        together_client = Together(api_key=api_key)
        logger.info("Together AI client initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing Together AI: {e}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'together_ai_ready': together_client is not None,
        'device': str(device) if device else None
    })

@app.route('/api/predict', methods=['POST'])
def predict_emotion():
    """Predict emotion from uploaded image"""
    try:
        if model is None:
            return jsonify({'error': 'Model not initialized'}), 500
            
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
            
        # Read and process image (convert to RGB as expected by your model)
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("L").convert("RGB")
        
        # Transform image and convert to float16 to match model precision
        image_tensor = transform(image).unsqueeze(0).to(device)
        image_tensor = image_tensor.half()  # Convert to float16
        
        # Predict using your model
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            predicted_emotion = class_names[predicted.item()]
            confidence_score = confidence.item()
            
        logger.info(f"Predicted emotion: {predicted_emotion} ({confidence_score:.2f})")
        
        return jsonify({
            'emotion': predicted_emotion,
            'confidence': confidence_score,
            'all_predictions': {
                class_names[i]: float(probs[0][i]) 
                for i in range(len(class_names))
            }
        })
        
    except Exception as e:
        logger.error(f"Error in emotion prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chatbot', methods=['POST'])
def chatbot_response():
    """Get chatbot response based on user message and emotion (integrating your chatbot.py logic)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        user_message = data.get('message', '')
        detected_emotion = data.get('emotion', 'neutral')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
            
        logger.info(f"Chatbot request - Message: {user_message}, Emotion: {detected_emotion}")
        
        # Use Together AI if available, otherwise fallback
        if together_client:
            try:
                # Create chat messages using the same approach as your chatbot.py
                chat_messages = [
                    {
                        "role": "system", 
                        "content": f"You are a helpful therapist, assisting people based on their emotion. The user is {detected_emotion}."
                    },
                    {
                        "role": "user", 
                        "content": user_message
                    }
                ]
                
                # Get response from Together AI using your existing configuration
                completion = together_client.chat.completions.create(
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                    messages=chat_messages,
                    max_tokens=1000,
                    temperature=0.7,
                    top_p=0.9,
                )
                
                response_text = completion.choices[0].message.content
                logger.info("Chatbot response generated successfully via Together AI")
                
            except Exception as ai_error:
                logger.error(f"Together AI error: {ai_error}")
                # Fallback response similar to your chatbot.py
                response_text = f"I understand you're feeling {detected_emotion}. That's a valid emotion, and I'm here to listen. Can you tell me more about what's on your mind?"
        else:
            # Fallback response when Together AI is not available
            response_text = f"I understand you're feeling {detected_emotion}. That's a valid emotion, and I'm here to listen. Can you tell me more about what's on your mind?"
        
        return jsonify({
            'reply': response_text,
            'emotion_context': detected_emotion
        })
        
    except Exception as e:
        logger.error(f"Error in chatbot response: {e}")
        # Always return a helpful fallback response
        emotion = data.get('emotion', 'neutral') if data else 'neutral'
        return jsonify({
            'reply': f"I understand you're feeling {emotion}. I'm here to support you. Could you tell me more about what's on your mind?",
            'error': str(e)
        }), 200

@app.route('/api/webcam/start', methods=['POST'])
def start_webcam():
    """Start webcam session (placeholder for future real-time features)"""
    return jsonify({'status': 'webcam session started'})

@app.route('/api/webcam/stop', methods=['POST'])
def stop_webcam():
    """Stop webcam session (placeholder for future real-time features)"""
    return jsonify({'status': 'webcam session stopped'})

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify server is working"""
    return jsonify({'status': 'Server is working', 'message': 'Test endpoint successful'})

@app.route('/api/upload-test', methods=['POST', 'OPTIONS'])
def upload_test():
    """Simple upload test endpoint"""
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    logger.info("=== UPLOAD TEST REQUEST RECEIVED ===")
    return jsonify({'status': 'Upload test successful', 'message': 'File upload endpoint is working'})

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Upload and analyze emotion from static image"""
    logger.info("=== UPLOAD REQUEST RECEIVED ===")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request headers: {dict(request.headers)}")
    logger.info(f"Request files: {list(request.files.keys())}")
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
            
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        # Read and process image (same as image_url.py logic)
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("L").convert("RGB")
        
        # Transform image and convert to float16 to match model precision
        image_tensor = transform(image).unsqueeze(0).to(device)
        image_tensor = image_tensor.half()  # Convert to float16
        
        # Predict using your model
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            predicted_emotion = class_names[predicted.item()]
            confidence_score = confidence.item()
            
        logger.info(f"Image upload - Predicted emotion: {predicted_emotion} ({confidence_score:.2f})")
        
        return jsonify({
            'emotion': predicted_emotion,
            'confidence': confidence_score,
            'all_predictions': {
                class_names[i]: float(probs[0][i]) 
                for i in range(len(class_names))
            }
        })
        
    except Exception as e:
        logger.error(f"Error in image upload prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting DeepMood API server...")
    
    # Debug: Print all registered routes
    logger.info("Registered routes:")
    for rule in app.url_map.iter_rules():
        logger.info(f"  {rule.rule} -> {rule.endpoint} [{', '.join(rule.methods)}]")
    
    # Initialize components
    model_ready = initialize_model()
    ai_ready = initialize_together_ai()
    
    if not model_ready:
        logger.warning("Model initialization failed - emotion prediction will not work")
    if not ai_ready:
        logger.warning("Together AI initialization failed - chatbot will use fallback responses")
    
    # Start server
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)