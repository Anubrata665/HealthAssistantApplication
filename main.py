# pip install flask transformers torch sentencepiece
import logging
import re
from flask import Flask, request, jsonify, render_template
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Set up logging to both console and file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chatbot.log')
    ]
)

app = Flask(__name__)

# Enable CORS for local development (in case browser blocks requests)
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    return response

# Load the saved model and base tokenizer
try:
    model = T5ForConditionalGeneration.from_pretrained("chatbot_model")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")  # Use base t5-small tokenizer
    device = model.device
    logging.info("Model and tokenizer loaded successfully")
except Exception as e:
    logging.error(f"Failed to load model or tokenizer: {str(e)}")
    raise

# Clean the text by removing unwanted characters
def clean_text(text):
    text = re.sub(r'\r\n', ' ', text)  # Remove carriage returns and line breaks
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'<.*?>', '', text)  # Remove any XML tags
    text = text.strip().lower()  # Strip and convert to lower case
    return text

# Chatbot function
def chatbot(dialogue):
    try:
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
        logging.debug(f"Chatbot response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error in chatbot function: {str(e)}")
        return "Sorry, I encountered an error. Please try again."

# Rendering Index root
@app.route("/")
def index():
    return render_template("index.html")

# API endpoint
@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Log entire request
        logging.debug(f"Full request: {request}")
        logging.debug(f"Request headers: {request.headers}")
        logging.debug(f"Request JSON: {request.json}")

        # Get user message and handle encoding
        user_message = request.json.get("message", "")
        if not isinstance(user_message, str):
            logging.warning(f"User message is not a string: {user_message}")
            user_message = str(user_message)
        user_message = user_message.encode('ascii', 'ignore').decode('ascii')
        logging.debug(f"Received user_message: '{user_message}' (type: {type(user_message)})")

        if not user_message:
            logging.error("No message provided in request")
            return jsonify({"error": "Message is required"}), 400

        # Ultra-simplified case-insensitive check for "cure"
        lowered_message = user_message.lower().strip()
        logging.debug(f"Lowered message: '{lowered_message}'")
        # Split into words to avoid substring issues
        words = lowered_message.split()
        cure_detected = "cure" in words
        logging.debug(f"Words in message: {words}")
        logging.debug(f"Cure detected: {cure_detected} (checked 'cure' in words: {words})")

        if cure_detected:
            logging.info("Detected 'cure' in message (case-insensitive)")
            response = "I Recommend You To Consult A Doctor"
        else:
            logging.info("No 'cure' detected, calling chatbot")
            response = chatbot(user_message)

        logging.debug(f"Returning response: '{response}'")
        return jsonify({"response": response})

    except Exception as e:
        logging.error(f"Error in /chat endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True)