from vllm import LLM
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize LLaMA 3 model
model = LLM("segolilylabs/Lily-Cybersecurity-7B-v0.2", quantize=True)  # Adjust the path and options as needed

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt")
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    # Generate response using the model
    response = model.generate(prompt)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
