from llama_cpp import Llama
import random
import os
from dotenv import load_dotenv
from akashic.utils import count_tokens

load_dotenv()

default_system_message = os.environ['SYSTEM_MESSAGE']
default_context_length = 4096
path_to_llama_cpp = os.environ['LLAMA_CPP_PATH'] 

class AkashicModel():
    def __init__(self, model_path, context_length=default_context_length, format="llama_2", seed=2023):
        self.model = Llama(
            model_path=f"{path_to_llama_cpp}/{model_path}",
            verbose=False,
            n_gpu_layers=-1, 
            seed=seed, 
            n_ctx=context_length, chat_format=format
        )
    def prompt(self, messages, stream):
        return self.model.create_chat_completion(messages, stream=stream)

class AkashicAgent:
    def __init__(self, model, context_length=default_context_length, system_message=default_system_message):
        self.model = model
        self.context_length = context_length
        self.messages = []
        self.system_message = system_message
    
    def add_message(self, role, content, prepend=False):
        if prepend:
            self.messages = [{"role": role, "content": content}] + self.messages
        else:
            self.messages.append({"role": role, "content": content})
    
    def truncate_messages(self):
        token_limit = self.context_length - count_tokens(self.system_message)

        messages_copy = self.messages[:]
        
        total_tokens = sum(count_tokens(msg['content']) for msg in messages_copy)
        while total_tokens > token_limit:
            messages_copy.pop(0)
            total_tokens = sum(count_tokens(msg['content']) for msg in messages_copy)
        
        return [{"role": 'system', "content": self.system_message}] + messages_copy
    
    def prepare_prompt(self, user_input):
        self.add_message("user", user_input)
        return self.truncate_messages()
    
    def send_prompt(self, user_input, stream=True):
        prompt = self.prepare_prompt(user_input)

        if stream:
            answers = self.model.prompt(prompt, stream=stream)

            response = ""
            
            for i in answers:
                if 'content' in i['choices'][0]['delta'].keys():
                    chunk = i['choices'][0]['delta']['content']
                    response += chunk
                    yield chunk

        else:
            answers = self.model.prompt(prompt, stream=stream)

            response = answers['choices'][0]['message']['content']

            yield response

        self.add_message('assistant', response)