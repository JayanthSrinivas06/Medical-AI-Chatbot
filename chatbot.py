import streamlit as st
import pickle
import numpy as np
from pathlib import Path
import random
import time

# ML/NLP Imports
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- 1. SETUP & CONFIGURATION ---

@st.cache_resource
def load_nltk_data():
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)

load_nltk_data()

# --- 2. LOAD MODEL & PREPROCESSORS ---

@st.cache_resource
def load_chatbot_artifacts():
    """Loads the trained model, tokenizer, label encoder, and responses."""
    OUT_DIR = Path("chatbot_model_output")
    MODEL_PATH = OUT_DIR / "medical_chatbot_model.h5"
    TOKENIZER_PATH = OUT_DIR / "tokenizer.pkl"
    LABELENC_PATH = OUT_DIR / "label_encoder.pkl"
    RESP_PATH = OUT_DIR / "responses_by_tag.pkl"

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(LABELENC_PATH, "rb") as f:
        le = pickle.load(f)
    with open(RESP_PATH, "rb") as f:
        responses_by_tag = pickle.load(f)
    
    return model, tokenizer, le, responses_by_tag

model, tokenizer, le, responses_by_tag = load_chatbot_artifacts()

# Get model parameters
MAX_LEN = model.input_shape[0][1]
BOW_FEATURES = model.input_shape[1][1]
CONF_THRESHOLD = 0.65

# --- 3. CHATBOT LOGIC (WITH NEW FORMATTING) ---

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum()]
    return " ".join(tokens)

def text_to_multihot(text):
    arr = np.zeros((1, BOW_FEATURES), dtype=np.float32)
    for word in text.split():
        idx = tokenizer.word_index.get(word)
        if idx and idx < BOW_FEATURES:
            arr[0, idx - 1] = 1.0
    return arr

def predict_intent(text):
    if not text: return None, 0.0
    processed_text = preprocess_text(text)
    if not processed_text: return None, 0.0
    seq = tokenizer.texts_to_sequences([processed_text])
    seq_pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    bow = text_to_multihot(processed_text)
    probs = model.predict({"seq_input": seq_pad, "bow_input": bow}, verbose=0)[0]
    top_idx = probs.argmax()
    top_prob = probs[top_idx]
    pred_tag = le.inverse_transform([top_idx])[0]
    return pred_tag, top_prob

def get_response(user_input):
    """
    Predicts intent and returns all formatted responses in a single message.
    """
    predicted_tag, confidence = predict_intent(user_input)

    # Handle conversational intents
    if predicted_tag in {"greeting", "goodbye", "thanks"} and confidence > CONF_THRESHOLD:
        return random.choice(responses_by_tag.get(predicted_tag))

    # Handle medical intents
    if confidence > CONF_THRESHOLD:
        responses_list = responses_by_tag.get(predicted_tag, [])
        
        if not responses_list or not any(s.strip() for s in responses_list):
            return f"I recognize the topic is '{predicted_tag}', but I don't have specific advice for it yet. Please consult a doctor."
        
        # --- ENHANCED FORMATTING LOGIC ---
        if len(responses_list) == 1:
            return responses_list[0]
        else:
            # Create a bold heading
            heading = f"**Here are the first-aid steps for {predicted_tag}:**\n"
            # Create a bulleted list from the responses
            bullet_points = "\n".join(f"* {resp}" for resp in responses_list)
            return heading + bullet_points
        # --- END OF CHANGE ---

    # Handle low confidence
    else:
        return "I'm not quite sure how to help with that. Could you please rephrase? If this is an emergency, please seek professional medical help."

# --- 4. STREAMLIT UI ---

st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")

st.title("🩺 Medical First-Aid Chatbot")
st.caption("Your AI assistant for basic first-aid questions. This is not a substitute for professional medical advice.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with first-aid today?"}]

# Display chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Generator for the typing effect
def stream_response(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.04)

# Handle user input
if prompt := st.chat_input("Enter your symptoms or question..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the full, formatted response and display it
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(prompt)
        
        # This line fixes the formatting by using st.markdown
        st.markdown(response) 
    
    # Add the complete assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})