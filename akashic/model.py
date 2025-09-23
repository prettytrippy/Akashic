from llama_cpp import Llama
import os
from dotenv import load_dotenv
from akashic.utils import count_tokens

load_dotenv()

path_to_llama_cpp = os.environ['LLAMA_CPP_PATH'] 

class AkashicModel():
    def __init__(self, model_path, context_length=1024, format="llama-3", seed=2023):
        self.model = Llama(
            model_path=f"{path_to_llama_cpp}/{model_path}",
            verbose=False,
            n_gpu_layers=-1, 
            seed=seed, 
            n_ctx=context_length, chat_format=format
        )
        self.context_length = context_length

    def __len__(self):
        return self.context_length
    
    def truncate_messages(self, messages):
        # Yeah, definitely fix this shit
        token_limit = int(len(self) * 0.8)

        messages_copy = messages
        
        total_tokens = sum(count_tokens(msg['content']) for msg in messages_copy)

        while total_tokens >= token_limit:
            messages_copy.pop(0)
            total_tokens = sum(count_tokens(msg['content']) for msg in messages_copy)
  

        return messages_copy
    
    def raw(self, text, max_tokens=None):
        return self.model(text, max_tokens=max_tokens, stream=False)

    def chat(self, messages, stream=False, max_tokens=-1, top_p=0.95, temperature=0.2):
        messages = self.truncate_messages(messages)
        answer = self.model.create_chat_completion(messages, stream=stream, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        if stream:
            for i in answer:
                if 'content' in i['choices'][0]['delta'].keys():
                    chunk = i['choices'][0]['delta']['content']
                    yield chunk
        else:
            yield answer['choices'][0]['message']['content']

    def complete(self, text, stream, max_tokens=None):
        output = self.model(
            text, max_tokens=max_tokens, stream=stream
        )
        if stream:
            for i in output:
                yield i['choices'][0]['text']
        else:
            yield output['choices'][0]['text']