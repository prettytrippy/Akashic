import fitz
import re
import numpy as np
import json
from tqdm import tqdm
from akashic.utils import count_tokens, embed_text, read_file

class AkashicRecord():
    def __init__(self, filename, path, max_tokens):
        self.filename = filename
        self.path = path
        self.max_tokens = max_tokens
        self.files = []
        self.embeddings = np.array([])

    def jsonify(self):
        embeddings = json.dumps(self.embeddings.tolist())
        files = json.dumps(self.files)
        return f'{{"Filename":"{self.filename}", "Path":"{self.path}", "Max Tokens":"{self.max_tokens}", "Files":{files}, "Embeddings":"{embeddings}"}}'

    def dejsonify(self, obj):
        self.filename = obj['Filename']
        self.path = obj['Path']
        self.max_tokens = int(obj['Max Tokens'])
        self.files = obj['Files']
        self.embeddings = np.array(json.loads(obj['Embeddings']), dtype=float)

    def get_embedding(self):
        embedding = np.mean(self.embeddings, axis=0)
        return embedding / np.linalg.norm(embedding)
    
    def rank_files(self, vec):
        vec /= np.linalg.norm(vec)
        sims = np.dot(vec, self.embeddings.T)
        idxs = np.argsort(sims)
        ranked_files = np.array(self.files)
        return ranked_files[idxs][::-1], sims[idxs][::-1]
    
    def chunk(self):
        txt = read_file(self.filename)
        txts = tokenize(txt)

        running_list = []
        running_string = ""
        for i in txts:
            if count_tokens(i) + count_tokens(running_string) < self.max_tokens:
                running_string += ' ' + i
            else:
                running_list.append(running_string)
                running_string = i
        running_list.append(running_string)
        return running_list
    
    def write_file(self, name, txt):
        with open(f"{self.path}/{name}", 'w') as file:
            file.write(txt)

    def upload_chunk(self, chunk, idx):
        embedding = embed_text(chunk)
        filename = f"{self.filename}.{idx}.akashic.txt"
        self.files.append(filename)
        if len(self.embeddings) == 0:
            self.embeddings = np.array([embedding])
        else:
            self.embeddings = np.append(self.embeddings, [embedding], axis=0)

        self.write_file(filename, chunk)

    def upload_all(self):
        chunks = list(self.chunk())
        for idx in tqdm(range(len(chunks)), desc=f"Uploading file {self.filename}"):
            self.upload_chunk(chunks[idx], idx)
        self.embeddings /= np.linalg.norm(self.embeddings, keepdims=True, axis=-1)