from flask import Flask, request, jsonify, Response, stream_with_context
from llama_cpp import Llama
import random
import json
import time
from akashic.agent import AkashicAgent
from dotenv import load_dotenv
import os

load_dotenv()

model_path = os.environ['MODEL_PATH'] 
model_format = os.environ['MODEL_FORMAT']
collections_directory = os.environ['COLLECTIONS']
context_length = 8192

chatter = AkashicAgent(model_path, context_length=context_length, format=model_format)

start_token = "<|endoftext|>"
end_token = "<|endoftext|>"
start_token = "<s>"
end_token = "</s>"

model = Llama(
    model_path=f"{path_to_llama_cpp}/{model_path}",
    verbose=True,
    n_gpu_layers=-1, 
    seed=random.randint(1000, 9999), 
    n_ctx=1<<10, chat_format='llama-2'
)

def shorten_prompt(prompt):
    # get rid of <fim_prefix> with the 12s
    prompt = prompt.split("<fim_suffix>")
    return prompt[0][12:]

app = Flask(__name__)

@app.route('/completion', methods=['POST'])
def completion():
    time.sleep(1)
    data = request.get_json()

    prompt = data['prompt']
    prompt = shorten_prompt(prompt)
    print("\nPrompt:", prompt)
    stream = data.get('stream', True)
    n_predict = 8

    answers = model.create_completion(prompt, stream=stream, max_tokens=n_predict)
    answers = list(i['choices'][0]['text'] for i in answers)

    print("\nResponse:", "".join(answers))


    if stream:
        def generate():
            for answer in answers:
                partial_response = {"content": answer}
                yield f"data: {json.dumps(partial_response)}\n\n"
            yield "data: [DONE]\n\n"

        return Response(stream_with_context(generate()), content_type='text/event-stream')
    else:
        full_answer = "".join(answers)
        final_response = {"content": full_answer}
        return jsonify(final_response)

if __name__ == '__main__':
    app.run(port=5000, debug=False)

