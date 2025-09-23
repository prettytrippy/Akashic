from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import time
from pprint import pprint

from akashic.model import AkashicModel
from akashic.utils import count_tokens

load_dotenv()

model_path = os.environ['MODEL_PATH'] 
model_format = os.environ['MODEL_FORMAT']
context_length = 9000

model = AkashicModel(model_path, context_length, format=model_format)

app = Flask(__name__)

@app.route('/chat/completions', methods=['POST'])
def chat():
    data = request.json
    
    messages = data.get('messages', [])
    # temperature = data.get('temperature', 0.2)
    # top_p = data.get('top_p', 0.95)

    prompt_tokens = sum([count_tokens(v) for message in messages for v in message.values()])

    try:
        # response_string = list(model.chat(messages, stream=False, temperature=temperature, top_p=top_p))[0]
        response_string = list(model.chat(messages, stream=False))[0]
        completion_tokens = count_tokens(response_string)
        total_tokens = prompt_tokens + completion_tokens
        response_dict = {
            "id": "ego",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "custom_server/none_of_your_business",
            "choices": [
                {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_string
                },
                "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }
        print(response_string)
        return jsonify(response_dict), 200
    
    except Exception as e:
        print("Error:", e)
        response_dict = {"error":str(e)}
        return jsonify(response_dict), 405

@app.route('/v1/model/info', methods=['GET'])
def info():
    response = "placeholder"
    return jsonify({'content': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8765, debug=False)

