from llama_cpp import Llama
import random

context_length = 1024

system_prompt = """You are a large language model,
who answers questions for humans.
Always be as helpful and honest as you can.
Always reply in markdown format."""

# change this to your absolute path to llama.cpp
path_to_llama_cpp = "/Users/trippdow/TrippUtilities/llama.cpp/models"

class AkashicChatbot():
    def __init__(self, model_path, system_prompt=system_prompt, context_length=context_length, format="llama", stop="[\INST]"):
        self.model = Llama(
            model_path=f"{path_to_llama_cpp}/{model_path}",
            verbose=False,
            n_gpu_layers=-1, 
            seed=random.randint(1000, 9999), 
            n_ctx=context_length, chat_format=format
        )

        self.system_prompt = system_prompt
        self.stop = [stop]
        self.messages = [{"role": "system", "content": system_prompt}]

    def chat_stream(self, user, context="", n=1, record=True):
        if context:
            user = f"{user}\nHere's some context that may help: {context}"
        self.messages.append({"role": "user", "content": user})

        answers = self.model.create_chat_completion(
            self.messages[-n*2-1:], stop = self.stop, stream = True
        )

        response = ""
        self.messages.append({"role": "assistant", "content": response})
        
        for i in answers:
            if 'content' in i['choices'][0]['delta'].keys():
                chunk = i['choices'][0]['delta']['content']
                response += chunk
                if record:
                    self.messages[-1]['content'] = response
                yield chunk

    def chat_text(self, user, context="", n=1, record=True):
        if context:
            user = f"{user}\nHere's some context that may help: {context}"
        self.messages.append({"role": "user", "content": user})

        answers = self.model.create_chat_completion(
            self.messages[-n*2-1:], stop = self.stop, stream = True
        )

        response = ""
        self.messages.append({"role": "assistant", "content": response})
        
        for i in answers:
            if 'content' in i['choices'][0]['delta'].keys():
                chunk = i['choices'][0]['delta']['content']
                response += chunk
                if record:
                    self.messages[-1]['content'] = response

        return response

    