from flask import Flask, request, jsonify, Response, stream_with_context
from akashic.model import AkashicModel
from akashic.agent import AkashicAgent
from dotenv import load_dotenv
import os
import fitz
from akashic.utils import count_tokens
import json
from pprint import pprint

load_dotenv()

model_path = os.environ['MODEL_PATH'] 
model_format = os.environ['MODEL_FORMAT']
context_length = 8192

code_model_path = os.environ['CODE_MODEL_PATH'] 
code_model_format = os.environ['CODE_MODEL_FORMAT']
code_context_length = 512

code_model = AkashicModel(code_model_path, context_length=code_context_length, format=code_model_format)
model = AkashicModel(model_path, context_length, format=model_format)
chatter = AkashicAgent(model)

app = Flask(__name__)

def format_prompt(prompt):
    prompt = prompt.split('<fim_suffix>')[0]
    return prompt[12:]

@app.route('/completion', methods=['POST'])
def completion():
    data = request.get_json()

    # pprint(data)

    prompt = data['prompt']
    stream = data.get("stream", False)
    n_predict = data.get("n_predict", 16)

    prompt = format_prompt(prompt)
    answers = code_model.complete(prompt, stream=stream, max_tokens=n_predict)

    if stream:
        def generate():
            for answer in answers:
                partial_response = {"content": answer}
                yield jsonify(partial_response)

        return Response(stream_with_context(generate()), content_type='text/event-stream')
    else:
        full_answer = "".join(answers)
        final_response = {"content": full_answer}
        return jsonify(final_response)
    
def cut_context(txt, length):
    tokens = count_tokens(txt)
    proportion = int(len(tokens)/length)
    txt = txt.split()
    return " ".join(txt[-len(txt)//(proportion+1):])

def get_context(document, page_num, pages=4):
    byte_array = bytearray(document)
    path = "reconstructed_file.pdf"
    with open(path, "wb") as pdf_file:
        pdf_file.write(byte_array)

    text = ""
    pdf_doc = fitz.open(path)

    start, end = page_num - pages, page_num
    start, end = max(0, start), min(end, pdf_doc.page_count)
    for i in range(start, end):
        page = pdf_doc.load_page(i)
        text += page.get_text()
    pdf_doc.close()
    return cut_context(text, chatter.context_length//2)

def process_text(question, context):
    txt = f"""
    Please help answer this question about an academic text: {question}.\n
    Here's some context from the text in question, which may or may not help: {context}.
    """
    answers = chatter.send_prompt(txt, stream=True)
    response = ""
    for chunk in answers:
        response += chunk
        print(chunk, end="", flush=True)

    return f"\t\n\r\n\n{response}"

@app.route('/process', methods=['POST'])
def process():
    print("AZFKLJSHFOSLFIHJAPLFKJHSLFKHDSOIRHJNKXMKC ")
    # 'text', 'page_number', 'pdf_file_data'
    data = request.json
    
    query = data.get('text')
    document = data.get('pdf_file_data')
    page = int(data.get('page_number'))

    print("DAta:", data, query, document, page)

    context = get_context(document, page)
    output_text = process_text(query, context)
    return jsonify({'content': output_text})

if __name__ == '__main__':
    app.run(port=8765, debug=False)

