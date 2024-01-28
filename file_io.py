import fitz
from text_utilities import filename_split

def pdf_to_text(pdf_path):
    text = ''
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    pdf_document.close()
    return text

def read_file(path):
    filename, extension = filename_split(path)
    if extension == 'pdf':
        return pdf_to_text(path), filename
    else:
        return open(path, 'r', encoding='utf-8', errors='ignore').read(), filename
    
def write_file_bytes(filename, bytes):
    with open(filename, 'wb') as file:
        file.write(bytes)

def write_file_text(filename, txt):
    with open(filename, 'w') as file:
        file.write(txt)