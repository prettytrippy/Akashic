from llama_cpp import Llama
import os
from dotenv import load_dotenv

load_dotenv()

path_to_llama_cpp = os.environ['LLAMA_CPP_PATH'] 

class AkashicModel():
    def __init__(self, model_path, context_length=1024, format="llama_2", seed=2023):
        self.model = Llama(
            model_path=f"{path_to_llama_cpp}/{model_path}",
            verbose=False,
            n_gpu_layers=-1, 
            seed=seed, 
            n_ctx=context_length, chat_format=format
        )

    def __len__(self):
        return self.model.n_ctx

    def raw(self, text, max_tokens=None):
        return self.model(text, max_tokens=max_tokens, stream=False)

    def chat(self, messages, stream, max_tokens=-1):
        answer = self.model.create_chat_completion(messages, stream=stream, max_tokens=max_tokens)
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