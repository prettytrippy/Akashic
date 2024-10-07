from flask import Flask, render_template, request, redirect, url_for
from markdown import markdown
from markupsafe import Markup
from werkzeug.utils import secure_filename
from akashic.agents.chatbot import AkashicChatbot
from akashic.model import AkashicModel
from akashic.retrieval.retriever import AkashicRetriever
from akashic.utils import webify_messages
from dotenv import load_dotenv
import os

load_dotenv()

model_path = os.environ['MODEL_PATH'] 
model_format = os.environ['MODEL_FORMAT']
collections_directory = os.environ['COLLECTIONS']
context_length = 8192

model = AkashicModel(model_path, context_length=context_length, format=model_format)
chatter = AkashicChatbot(model)
archivist = AkashicRetriever(collections_directory, context_length//8)

current_collections = []

app = Flask(__name__)
# Markdown(app)

@app.template_filter('markdown')
def markdown_filter(text):
    return Markup(markdown(text))

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
    return render_template('collections.html', collections=archivist.collections.items())

@app.route("/delete_file_from_collection", methods=['POST'])
def delete_file_from_collection():
    if request.method == 'POST':
        folder_name = request.form['folder_name']
        file_name = request.form['file_name']
        archivist.remove_file(folder_name, file_name)
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
        archivist.add_file(folder_name, filename)

        if os.path.exists(filename):
            os.remove(filename)
            
        return redirect(url_for('collections'))
    
@app.route("/delete_collection", methods=['POST'])
def delete_collection():
    if request.method == 'POST':
        collection_name = request.form['collection_name']
        archivist.remove_collection(collection_name)
        return redirect(url_for('collections'))

@app.route("/add_collection", methods=['POST'])
def add_collection():
    if request.method == 'POST':
        collection_name = request.form['collection_name']
        archivist.create_collection(collection_name)
        return redirect(url_for('collections'))
    
@app.route("/chat", methods=['POST', 'GET'])
def chat():
    if request.method == 'POST':
        query = request.form.get('user_input')
        if query:
            # get any useful context from the document store, then prompt the chatbot
            hyde_document = list(chatter.send_prompt(query, stream=False))[0]
            chatter.messages = chatter.messages[:-2] # clear last two messages

            print("HYDE:", hyde_document)
            context = archivist.get_context(current_collections, hyde_document, n=3)
            query = f"{query}\nHere's some context that might help: {context}" if context else query
            stream = chatter.send_prompt(query, stream=True)
        
        for i in stream:
            print(i, end="", flush=True)
    
    return render_template('chat.html', collection_names=archivist.collections, messages=webify_messages(chatter.messages))

@app.route('/update_collections', methods=['POST'])
def update_collections():
    selected_collections = request.form.getlist('collections')

    global current_collections
    current_collections = selected_collections
    return redirect(url_for('chat'))

if __name__ == '__main__':
    app.run(debug=True)

