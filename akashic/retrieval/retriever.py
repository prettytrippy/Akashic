import numpy as np
from akashic.utils import *
from akashic.retrieval.record import AkashicRecord
import shutil
import os
import json
from glob import glob

class AkashicRetriever():
    def __init__(self, path, max_tokens):
        self.path = path
        self.max_tokens= max_tokens
        self.collections = dict()
        self.init_collections()

    def init_collections(self):
        collection_names = [name for name in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, name))]
        for collection in collection_names:
            full_collection_path = f"{self.path}/{collection}"
            filenames = glob(f"{full_collection_path}/*.akashic.txt")
            real_names = [self.get_original_filename(filename) for filename in filenames]
            self.collections[collection] = list(set(real_names))

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
        path = path.replace(f"{self.path}", "")
        chunks = path.split(".")
        path = ".".join(chunks[:-1])
        path = path.split("/")[-1]
        return path

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
            if f"{self.path}/{collection}/{self.get_original_filename(stored_filepath)}" == filepath_to_match:
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
    
    def get_context(self, collections, txt, n=3):
        files = self.rank_files(collections, txt, n=n)
        context = ""
        for filepath in files:
            with open(filepath, 'r') as file:
                context = f"{context}\n{file.read()}"
        return context