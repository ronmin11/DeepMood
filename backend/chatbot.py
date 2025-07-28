from together import Together
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

predicted_emotion = 'sad' #Put prediction of CNN here
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"  
user_input = "I am feeling very down today."  #user input via frontend typing will go here 

# Get API key from environment variables
api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    raise ValueError("TOGETHER_API_KEY not found in environment variables. Please check your .env file.")

client = Together(api_key=api_key)



chat = [
    {"role": 'system', 'content': f'You are a helpful therapist, assisting people based on their emotion. The user is {predicted_emotion}.'},
    {"role": 'user', 'content': user_input},
]

try:
    completion = client.chat.completions.create(
        model=MODEL,
        messages=chat,
        max_tokens=1000,
        temperature=0.7,
        top_p=0.9,
    )
    
    # For now, use a simple response
    response = f"I understand you're feeling {predicted_emotion}. That's a valid emotion, and I'm here to listen. Can you tell me more about what's on your mind?"
    print(response)
    
except Exception as e:
    print(f"Error calling Together AI: {e}")
    print("I'm sorry, I couldn't generate a response right now. Please try again.")