import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline

predicted_emotion = 'sad' #Put prediction of CNN here
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  
user_input = "I am feeling very down today."  #user input via frontend typing will go here 

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype=torch.bfloat16)

chat = [
    {"role": 'system', 'content': f'You are a helpful therapist, assisting people based on their emotion. The user is {predicted_emotion}.'},
    {"role": 'user', 'content': user_input},
]

tokenized_chat = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(tokenized_chat[0]))
outputs = model.generate(tokenized_chat, max_new_tokens=128) 

print(tokenizer.decode(outputs[0]))

