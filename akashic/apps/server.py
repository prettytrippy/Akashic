from flask import Flask, request, jsonify, Response, stream_with_context
from akashic.model import AkashicModel
from dotenv import load_dotenv
import os

load_dotenv()

model_path = os.environ['MODEL_PATH'] 
model_format = os.environ['MODEL_FORMAT']
context_length = 8192

code_model_path = os.environ['CODE_MODEL_PATH'] 
code_model_format = os.environ['CODE_MODEL_FORMAT']
code_context_length = 1024

# chatter = AkashicAgent(model_path, context_length=context_length, format=model_format)
code_model = AkashicModel(code_model_path, context_length=code_context_length, format=code_model_format)

app = Flask(__name__)

def format_prompt(prompt):
    prompt = prompt.split('<fim_suffix>')[0]
    return prompt[12:]

@app.route('/completion', methods=['POST'])
def completion():
    data = request.get_json()

    prompt = data['prompt']
    prompt = format_prompt(prompt)
    n_predict = 16

    answer = code_model.complete(prompt, stream=False, max_tokens=n_predict)
    print("\nANSWER:", "".join(list(answer)))

    # if stream:
    #     def generate():
    #         for chunk in answer:
    #             partial_response = {"content": chunk}
    #             yield f"data: {json.dumps(partial_response)}\n\n"
    #         yield "data: [DONE]\n\n"

    #     return Response(stream_with_context(generate()), content_type='text/event-stream')
    # else:

    final_response = {"content": answer}
    return jsonify(final_response)

if __name__ == '__main__':
    app.run(port=8080, debug=False)

