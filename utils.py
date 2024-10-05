import transformers
from sentence_transformers import SentenceTransformer
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

def count_tokens(txt):
    inputs = tokenizer(txt, return_tensors="pt").input_ids[0]
    return len(inputs)

def embed_text(txt):
    return model.encode(str(txt))

def webify_messages(messages):
    tuples = []
    for m in messages:
        if m['role'] != 'system':
            tuples.append((m['role'], m['content']))
    return tuples

