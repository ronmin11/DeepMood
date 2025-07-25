from together import Together
import getpass
import os

predicted_emotion = 'sad' #Put prediction of CNN here
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"  
user_input = "I am feeling very down today."  #user input via frontend typing will go here 
os.environ["TOGETHER_API_KEY"] = getpass.getpass("Enter Together API Key: ")
client = Together()



chat = [
    {"role": 'system', 'content': f'You are a helpful therapist, assisting people based on their emotion. The user is {predicted_emotion}.'},
    {"role": 'user', 'content': user_input},
]

completion = client.chat.completions.create(
    model=MODEL,
    messages=chat,
    max_tokens=1000,
    temperature=0.7,
    top_p=0.9,
)

print(completion.choices[0].message.content)