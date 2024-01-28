"""An embedder maps a string to a vector embedding"""

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def sentence_transformer_embedding(txt):
    return model.encode(str(txt))
    