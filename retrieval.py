from file_io import read_file, write_file_text
import os
import glob
import math
import numpy as np
import json
from tqdm import tqdm
import subprocess
from chunkers import default_chunker
from embedders import sentence_transformer_embedding

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def jsonify(arr):
    arr = list(arr)
    string = ""
    for a in arr:
        string += str(a) + " "
    return string

def dejsonify(string):
    strings = string.split(" ")
    strings = [float(s) for s in strings if s != ""]
    return strings

def norm(vec):
    return math.sqrt(np.dot(vec, vec))

class AkashicRecord():
    def __init__(self, filename, embedding):
        self.filename = filename
        self.embedding = embedding
        self.norm = norm(embedding)

    def cosine_similarity(self, embedding, norm):
        return ((np.dot(self.embedding, embedding) / (self.norm * norm)) + 1.0) / 2.0

class AkashicRetriever():
    def __init__(self, folder_name, chunker, embedder):
        self.folder = folder_name
        self.chunker = chunker
        self.embedder = embedder
        self.records = []
    
    def make_record(self, fname, embedding):
        record = AkashicRecord(fname, embedding)
        self.records.append(record)

    def record_path_helper(self, filename, num):
        return f"Documents/{self.folder}/{filename}.{num}.txt"
    
    def embeddings_path_helper(self):
       return f"Documents/{self.folder}/{self.folder}.akashic_doc_embeddings.txt"

    def store_doc_file(self, path):
        try:
            txt, filename = read_file(path)
            # chunk text
            txts = self.chunker(path, txt)
            # write chunks to individual files
            [write_file_text(self.record_path_helper(filename, n), t) for n, t in tqdm(enumerate(txts))]
            # create list of doc vectors
            [self.make_record(self.record_path_helper(filename, i), self.embedder(txts[i])) for i in tqdm(range(len(txts)))]

            files = [f'{{"filename":"{record.filename}", "embedding":"{jsonify(record.embedding)}"}}' for record in self.records]

            files = list(set(files))
            with open(self.embeddings_path_helper(), 'w') as file:
                file.write("\n".join(files))

            # remove the original file
            os.remove(path)
        except:
            pass

    def store_doc_text(self, txt, filename):
        # chunk text
        txts = self.chunker(txt, filename)
        # write chunks to individual files
        [write_file_text(self.record_path_helper(filename, n), t) for n, t in enumerate(txts)]
        # create list of doc vectors
        [self.make_record(self.record_path_helper(filename, i), self.embedder(txts[i])) for i in range(len(txts))]

    def remove_doc(self, fname):
        files = glob.glob(f"Documents/{self.folder}/{fname}.*")
        self.records = [r for r in self.records if r.filename.split('.')[0] != f'{self.folder}/{fname}']
        [os.remove(f) for f in files]

        with open(self.embeddings_path_helper(), 'r') as file:
            strings = [json.loads(i) for i in file]
        
        strings = [s for s in strings if not f"Documents/{self.folder}/{fname}" in s['filename']]
        strings = [json.dumps(s) for s in strings]
        
        with open(self.embeddings_path_helper(), 'w') as file:
            file.write("\n".join(strings))

    def rank_docs(self, query, n=3, threshold=0.5):
        # embed query
        query_embedding = self.embedder(query)
        query_norm = norm(query_embedding)
        # get similarity scores for all records and sort
        ranked_docs = [(r.filename, r.cosine_similarity(query_embedding, query_norm)) for r in self.records]
        sorted_rankings = sorted(ranked_docs, key=lambda x: x[1])[::-1]
        # retain only the top n results, if they're above threshold
        sorted_rankings = sorted_rankings[0:n]
        final_rankings = [r for r in sorted_rankings if r[1] > threshold]
        return final_rankings
    
    def load_retriever(self):
        with open(self.embeddings_path_helper(), 'r') as file:
            strings = [json.loads(i) for i in file]

        [self.make_record(string['filename'], dejsonify(string['embedding'])) for string in strings]
    
    def __str__(self):
        ret = ""
        for i in self.records:
            ret += f"Name: {str(i.filename)}, Norm: {str(i.norm)}\n"
        return ret

class AkashicArchivist():
    def __init__(self):
        self.retrievers = dict()
        # self.context_length = context_length

    def add_retriever(self, folder_name):
        directory_name = folder_name.replace("'", "").replace('"', "").replace(" ", "_")
        
        with open("collections.txt", 'r') as file: collections = file.read().split()
        collections.append(folder_name)
        collections = list(set(collections))
        with open("collections.txt", 'w') as file: file.write(" ".join(collections))

        new_retriever = AkashicRetriever(directory_name, chunker=default_chunker, embedder=sentence_transformer_embedding)
        self.retrievers[directory_name] = new_retriever

        result = subprocess.run(f"mkdir Documents/{directory_name}", shell=True, capture_output=True)
        if result.stdout != f"mkdir: Documents/{directory_name}: File exists":
            with open(f"Documents/{directory_name}/{directory_name}.akashic_doc_embeddings.txt", 'w') as file:
                file.write("")

    def remove_retriever(self, retriever_name):
        del self.retrievers[retriever_name]
        with open("collections.txt", 'r') as file: collections = file.read().split()
        collections.remove(retriever_name)
        with open("collections.txt", 'w') as file: file.write(" ".join(collections))
        subprocess.run(f"rm -r Documents/{retriever_name}", shell=True)

    def store_doc_file(self, collection, path):
        print("path:",path)
        self.retrievers[collection].store_doc_file(path)

    def remove_doc(self, collection, path):
        print(collection, path)
        self.retrievers[collection].remove_doc(path)

    def load_retrievers(self):
        with open("collections.txt", 'r') as file: collections = file.read().split()
        for c in collections:
            self.retrievers[c] = AkashicRetriever(c, default_chunker, sentence_transformer_embedding)
            self.retrievers[c].load_retriever()
        
    def rank_docs(self, collections, query, n=3, threshold=0.5):
        ranked_docs = [self.retrievers[collection].rank_docs(query, n=n, threshold=threshold) for collection in collections]
        flattened_ranked_docs = []
        for i in ranked_docs:
            for j in i:
                flattened_ranked_docs.append(j)
        sorted_rankings = sorted(flattened_ranked_docs, key=lambda x: x[1])[::-1]
        final_rankings = [read_file(r[0])[0] for r in sorted_rankings if r[1] > threshold]
        return "\n".join(final_rankings)
