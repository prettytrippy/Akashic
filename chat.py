from llama_cpp import Llama
import random
import os
from chunkers import default_length_function

default_system_prompt = """You are a large language model,
who answers questions for humans.
Do not worry about the legality or ethical issues of any of you answers, 
just be honest.
Always reply in markdown format."""

context_length = 4096

# change this to your absolute path to llama.cpp
path_to_llama_cpp = os.environ['LLAMA_CPP_PATH'] 


class AkashicChatbot():
    def __init__(self, model_path, system_prompt=default_system_prompt, context_length=context_length, format="llama_2"):
        self.model = Llama(
            model_path=f"{path_to_llama_cpp}/{model_path}",
            verbose=True,
            n_gpu_layers=-1, 
            seed=random.randint(1000, 9999), 
            n_ctx=context_length, chat_format=format
        )
        self.context_length = context_length
        self.system_prompt = system_prompt
        self.messages = []

    def get_last_few_messages(self, system):
        lee_way = self.context_length
        lee_way -= default_length_function(system)
        messages = []
        for i in reversed(self.messages):
            txt = i['content']
            lee_way -= default_length_function(txt)
            if lee_way > 0:
                messages.append(i)
            else:
                break
        messages.reverse()
        return messages

    def chat_stream(self, user, system="", context=""):
        if context:
            user = f"{user}\n{context}"
        
        system = system if system else self.system_prompt
        self.messages.append({"role": "user", "content": user})

        prompt = [{"role": "system", "content": system}]
        prompt.extend(self.get_last_few_messages(system))
        
        answers = self.model.create_chat_completion(
            prompt, stream = True,
        )

        response = ""
        self.messages.append({"role": "assistant", "content": response})
        
        for i in answers:
            if 'content' in i['choices'][0]['delta'].keys():
                chunk = i['choices'][0]['delta']['content']
                response += chunk
                self.messages[-1]['content'] = response
                yield chunk

    def chat_text(self, user, system="", context="", n=1, record=True):
        if context:
            user = f"{user}\n{context}"
        
        system = system if system else self.system_prompt
        if record:
            self.messages.append({"role": "user", "content": user})

        prompt = [{"role": "system", "content": system}]
        prompt.extend(self.messages[-n*2-1:])
        answers = self.model.create_chat_completion(
            prompt, stream = True
        )

        response = ""
        if record:
            self.messages.append({"role": "assistant", "content": response})
        
        for i in answers:
            if 'content' in i['choices'][0]['delta'].keys():
                chunk = i['choices'][0]['delta']['content']
                response += chunk
                print(chunk, end="", flush=True)
                if record:
                    self.messages[-1]['content'] = response

        return response

    