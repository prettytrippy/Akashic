import fitz
import re
import numpy as np
import json
from glob import glob
from utils import count_tokens, embed_text
import shutil
import os

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

class AkashicArchivist():
    def __init__(self, path, max_tokens):
        self.path = path
        self.max_tokens= max_tokens
        # If collections exist, fill a list with their names
        self.collections = dict()

    def create_collection(self, collection):
        if collection in self.collections.keys():
            raise ValueError(f"Collection {collection} already exists.")
        
        path = f"{self.path}/{collection}"
        try:
            os.makedirs(path, exist_ok=True)
            self.collections[collection] = []
        except Exception as e:
            print(f"Error creating directory: {e}")

    def remove_collection(self, collection):
        path = f"{self.path}/{collection}"
        if os.path.exists(path):
            shutil.rmtree(path)
            self.collections.pop(collection, None)
        else:
            print(f"Collection {collection} does not exist.")
       
    def get_original_filename(self, path):
        path = path.replace(".akashic.txt", "")
        chunks = path.split(".")
        return ".".join(chunks[:-1])

    def add_file(self, collection, filepath):
        record = AkashicRecord(filepath, f"{self.path}/{collection}", max_tokens=self.max_tokens)
        record.upload_all()
        self.add_to_masterfile(collection, record)
        self.collections[collection].append(filepath)

    def remove_file(self, collection, filepath):
        files_to_remove = []
        filepath_to_match = f"{self.path}/{collection}/{filepath}"
        filepaths = glob(f"{self.path}/{collection}/*.akashic.txt")

        for stored_filepath in filepaths:
            if self.get_original_filename(stored_filepath) == filepath_to_match:
                files_to_remove.append(stored_filepath)
        
        for file in files_to_remove:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error removing file {file}: {e}")
        
        self.remove_from_masterfile(collection, filepath)
        self.collections[collection].remove(filepath)

    def read_masterfile(self, collection):
        records = []
        masterfile_path = f"{self.path}/{collection}/akashic_{collection}_masterfile.txt"
        if not os.path.exists(masterfile_path):
            return []
        with open(masterfile_path, 'r') as file:
            for chunk in file:
                record = AkashicRecord("", "", 1)
                record.dejsonify(json.loads(chunk))
                records.append(record)
        return records

    def write_masterfile(self, collection, records):
        masterfile_path = f"{self.path}/{collection}/akashic_{collection}_masterfile.txt"
        with open(masterfile_path, 'w') as file:
            file.write("")
        with open(masterfile_path, 'a+') as file:
            for record in records:
                file.write(record.jsonify())
                file.write("\n")

    def add_to_masterfile(self, collection, record):
        records = self.read_masterfile(collection)
        records.append(record)
        self.write_masterfile(collection, records)

    def remove_from_masterfile(self, collection, filepath):
        records = self.read_masterfile(collection)
        records_to_keep = []
        for record in records:
            print("NAMES:", record.filename, filepath)
            if record.filename != filepath:
                records_to_keep.append(record)
        self.write_masterfile(collection, records_to_keep)

    def rank_files(self, collections, txt, n=3):
        embedding = embed_text(txt)
        embedding /= np.linalg.norm(embedding)

        similarity_dict = dict()

        for collection in collections:
            records = self.read_masterfile(collection)
            for record in records:
                filenames, similarities = record.rank_files(embedding)
                for filename, similarity in zip(filenames, similarities):
                    filename = f"{record.path}/{filename}"
                    similarity_dict[filename] = similarity
        
        sorted_keys = sorted(similarity_dict, key=similarity_dict.get)
        n = min(len(sorted_keys), n)
        best_files = sorted_keys[::-1]
        return best_files[:n]