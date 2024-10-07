
from akashic.model import AkashicModel
from akashic.agents.chatbot import AkashicChatbot
from dotenv import load_dotenv
import os
import random

load_dotenv()

if __name__ == "__main__":
    model_path = os.environ['MODEL_PATH'] 
    model_format = os.environ['MODEL_FORMAT']
    
    model = AkashicModel(model_path, context_length=8000, format=model_format, seed=random.randint(1000, 9999))
    chatter = AkashicChatbot(model)

    chatter.chat()