from flask import Flask, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import json

app = Flask(__name__)

# Load intents (FAQs) from a JSON file
intents_file = 'intent.json'
with open(intents_file, 'r') as file:
    intents = json.load(file)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocess and tokenize the data
def preprocess(sentence):
    return [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence)]

def bag_of_words(sentence, words):
    sentence_words = preprocess(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predict intent class and response
def predict(input_data):
    # Extract the list of unique words from the intents file
    words = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            words.extend(preprocess(pattern))
    words = sorted(set(words))  # Get unique words and sort them

    # Convert the input sentence to bag-of-words
    bow = bag_of_words(input_data, words)

    # Determine the intent based on the patterns
    predicted_intents = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # If the input is similar to the pattern, add the intent to the list
            if bow.tolist() == bag_of_words(pattern, words).tolist():
                predicted_intents.append(intent['responses'])

    # If a matching intent is found, return a random response; otherwise, return a default response
    if predicted_intents:
        return random.choice(predicted_intents[0])  # Return a random response from the matched intent
    else:
        return "I'm sorry, I didn't understand that."

# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.json
    input_data = data.get('message', '')  # Get the message from the incoming JSON
    response = predict(input_data)  # Call the predict function
    return jsonify({"response": response})  # Return the response as JSON

if __name__ == "__main__":
    app.run(port=5000)  # Run the Flask app on port 5000
