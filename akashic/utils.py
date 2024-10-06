import tiktoken
from sentence_transformers import SentenceTransformer
import os
import fitz

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

def pdf_to_text(pdf_path):
        text = ''
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        pdf_document.close()
        return text
    
def read_file(filename):
    extension = filename.split(".")[-1]
    if extension == 'pdf':
        return pdf_to_text(filename)
    else:
        return open(filename, 'r', encoding='utf-8', errors='ignore').read()
    
