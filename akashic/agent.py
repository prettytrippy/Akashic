import os
from dotenv import load_dotenv
from akashic.utils import count_tokens

load_dotenv()

default_system_message = os.environ['SYSTEM_MESSAGE']

class AkashicAgent:
    def __init__(self, model, system_message=default_system_message):
        self.model = model
        self.messages = []
        self.system_message = system_message

    def set_system_message(self, system):
        self.system_message = system

    def set_context_length(self, new_length):
        self.model.context_length = new_length

    def get_context_length(self):
        return self.model.context_length
    
    def add_message(self, role, content, prepend=False):
        if prepend:
            self.messages = [{"role": role, "content": content}] + self.messages
        else:
            self.messages.append({"role": role, "content": content})
    
    def truncate_messages(self):
        token_limit = self.model.context_length - count_tokens(self.system_message)

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
        result = self.model.chat(prompt, stream=stream)
        response = ""

        if stream:
            for i in result:
                response += i
                yield i
        else:
            response = result
            yield response

        self.add_message('assistant', response)