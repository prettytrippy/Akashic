import tiktoken
from sentence_transformers import SentenceTransformer
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(txt):
    inputs = encoding.encode(txt)
    return len(inputs)

def embed_text(txt):
    return model.encode(str(txt))

def webify_messages(messages):
    tuples = []
    for m in messages:
        if m['role'] != 'system':
            tuples.append((m['role'], m['content']))
    return tuples

