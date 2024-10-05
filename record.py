import fitz
import re
import numpy as np
import json
from utils import count_tokens, embed_text

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
    
    def pdf_to_text(pdf_path):
        text = ''
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        pdf_document.close()
        return text
    
    def read_file(self):
        extension = self.filename.split(".")[-1]
        if extension == 'pdf':
            return self.pdf_to_text(self.filename)
        else:
            return open(self.filename, 'r', encoding='utf-8', errors='ignore').read()
        
    def chunk(self):
        txt = self.read_file()
        pattern = r'\n|\t|\r'

        txts = re.split(pattern, txt)
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
        chunks = self.chunk()
        for idx, chunk in enumerate(chunks):
            self.upload_chunk(chunk, idx)
        self.embeddings /= np.linalg.norm(self.embeddings, keepdims=True, axis=-1)