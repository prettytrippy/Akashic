from flask import Flask, render_template, request, redirect, url_for
from flaskext.markdown import Markdown
from werkzeug.utils import secure_filename
from chatbots import *
from retrieval import AkashicArchivist
from text_utilities import collections_dict, webify_messages

context_length = 1024

archivist = AkashicArchivist()
archivist.load_retrievers()
chatter = llama3_Q4_chatbot(context_length=1024)

current_collections = []

app = Flask(__name__)
Markdown(app)

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        button_clicked = request.form.get('button_clicked')
        if button_clicked == 'collections':
            return redirect(url_for('collections'))
        elif button_clicked == 'chat':
            return redirect(url_for('chat'))
    return render_template('index.html')

@app.route("/collections")
def collections():
    collection_table = collections_dict()
    return render_template('collections.html', collections=collection_table.items())

@app.route("/delete_file_from_collection", methods=['POST'])
def delete_file_from_collection():
    if request.method == 'POST':
        folder_name = request.form['folder_name']
        file_name = request.form['file_name']
        archivist.remove_doc(folder_name, file_name)
        return redirect(url_for('collections'))

@app.route("/add_file_to_collection", methods=['POST'])
def add_file_to_collection():
    if request.method == 'POST':
        folder_name = request.form['folder_name']
        if 'file' not in request.files: return redirect(url_for('collections'))

        file = request.files['file']
        if file.filename == '': return redirect(url_for('collections'))

        filename = secure_filename(file.filename)
        file.save(filename)
        archivist.store_doc_file(folder_name, filename)
        return redirect(url_for('collections'))
    
@app.route("/delete_collection", methods=['POST'])
def delete_collection():
    if request.method == 'POST':
        collection_name = request.form['collection_name']
        archivist.remove_retriever(collection_name)
        return redirect(url_for('collections'))

@app.route("/add_collection", methods=['POST'])
def add_collection():
    if request.method == 'POST':
        collection_name = request.form['collection_name']
        archivist.add_retriever(collection_name)
        return redirect(url_for('collections'))
    
@app.route("/chat", methods=['POST', 'GET'])
def chat():
    collection_names = collections_dict().keys()

    if request.method == 'POST':
        query = request.form.get('user_input')
        if query:
            # get any useful context from the document store, then prompt the chatbot
            # hyde_document = chatter.chat_text(query, context="", n=1)
            hyde_document = query

            context = archivist.rank_docs(current_collections, hyde_document, 3, .5)
            stream = chatter.chat_stream(query, context=context)
        
        for i in stream:
            print(i, end="", flush=True)
    
    return render_template('chat.html', collection_names=collection_names, messages=webify_messages(chatter.messages))

@app.route('/update_collections', methods=['POST'])
def update_collections():
    selected_collections = request.form.getlist('collections')

    global current_collections
    current_collections = selected_collections
    print(current_collections)
    return redirect(url_for('chat'))

if __name__ == '__main__':
    app.run(debug=True)

