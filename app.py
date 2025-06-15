import sys
import io
import os
import re
from flask import Flask, render_template, request, jsonify

from diet_app.routes import diet_bp
from tumor_app.routes import tumor_bp

# Model dependencies
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Enable UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

# Add paths to submodules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'tumor_app')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'diet_app')))

app = Flask(__name__)

# Register Blueprints
app.register_blueprint(tumor_bp, url_prefix='/tumor')
app.register_blueprint(diet_bp, url_prefix='/diet')

# === Chatbot Setup === #
model = T5ForConditionalGeneration.from_pretrained("chatbot_model")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
device = model.device

def clean_text(text):
    text = re.sub(r'\r\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.strip().lower()
    return text

def chatbot(dialogue):
    dialogue = clean_text(dialogue)
    inputs = tokenizer(dialogue, return_tensors="pt", truncation=True, padding="max_length", max_length=250)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model.generate(
        inputs["input_ids"],
        max_length=250,
        num_beams=4,
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# === Routes === #

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    response = chatbot(user_message)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
