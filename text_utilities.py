import os

def filename_split(path):
    s = path.split(".")
    return ".".join(s[:-1]), s[-1]

def collections_helper(path):
    try:
        files_list = os.listdir(path)
    except OSError as e:
        return f"Error: {e}"
    real_files = set()
    for f in files_list:
        fs = f.split(".")
        if fs[1] != 'akashic_doc_embeddings':
            real_files |= {fs[0]}
    return list(real_files)

def collections_dict():
    with open("collections.txt", 'r') as file: 
        collections = file.read().split()
    
    ret_dict = dict()
    for c in collections:
        ret_dict[c] = collections_helper(f"Documents/{c}")
    return ret_dict

def webify_messages(messages):
    tuples = []
    for m in messages:
        if m['role'] != 'system':
            tuples.append((m['role'], m['content']))
    return tuples

