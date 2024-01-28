"""A chunker maps a string to a list of strings,
   in order to avoid context overflow when inputting documents to an LLM"""

import re
from text_utilities import filename_split
import transformers
from chat import context_length

tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

def default_length_function(txt):
    inputs = tokenizer(txt, return_tensors="pt").input_ids[0]
    return len(inputs)

default_pattern = r'\n|\t|\r'

default_max_length = context_length // 4

def chunk_doc_frame(txt, length_function, max_length, pattern):
    txts = re.split(pattern, txt)
    running_list = []
    running_string = ""
    for i in txts:
        if length_function(i) + length_function(running_string) < max_length:
            running_string += ' ' + i
        else:
            running_list.append(running_string)
            running_string = i
    running_list.append(running_string)
    return running_list

def chunker_factory(txt, pattern):
    return chunk_doc_frame(txt, default_length_function, default_max_length, pattern)

def default_chunker(path, txt):
    _, extension = filename_split(path)
    if extension == 'py':
        pattern = r'\r|\n'
    elif extension == 'c':
        pattern = r'}|{|\n|\r|;'
    elif extension == 'txt':
        pattern = r'\n|\r|. |\t'
    else:
        pattern = default_pattern
        
    return chunker_factory(txt, pattern)
