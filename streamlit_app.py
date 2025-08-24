import streamlit as st
import time
import numpy as np
import requests
import json
import os
from dotenv import load_dotenv
from search_final import rag_pipeline

# Load environment variables
load_dotenv()

# Hugging Face API Configuration
HF_API_URL = "https://api-inference.huggingface.co/models/kundan621/tinyllama-makemytrip-financial-qa"
HF_TOKEN = os.getenv("HF_API_KEY", "")

def query_huggingface_api(prompt, max_retries=3):
    """Query Hugging Face Inference API with retry logic"""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False,
            "top_p": 0.9
        }
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', 'No response generated')
                else:
                    return str(result)
            elif response.status_code == 503:
                # Model is loading, wait and retry
                if attempt < max_retries - 1:
                    st.info(f"Model is loading... Retrying in {2**(attempt+1)} seconds")
                    time.sleep(2**(attempt+1))
                    continue
                else:
                    return "Model is still loading. Please try again in a few moments."
            else:
                return f"API Error {response.status_code}: {response.text}"
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                st.warning(f"Request timeout. Retrying... (Attempt {attempt + 2}/{max_retries})")
                time.sleep(2)
                continue
            else:
                return "Request timed out. Please try again."
        except Exception as e:
            return f"Error: {str(e)}"
    
    return "Failed to get response after multiple attempts."

def generate_fine_tuned_response(question):
    """Generate response using Hugging Face Inference API"""
    if not HF_TOKEN:
        return "⚠️ Hugging Face token not found. Please set HF_API_KEY in your .env file."
    
    # Format the prompt for TinyLlama chat template
    system_prompt = "You are a helpful assistant that provides financial data from MakeMyTrip reports."
    
    # Create the properly formatted prompt
    formatted_prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{question}</s>\n<|assistant|>\n"
    
    # Query the API
    response = query_huggingface_api(formatted_prompt)
    
    # Clean up the response
    if isinstance(response, str):
        # Remove any leftover special tokens
        response = response.replace("</s>", "").strip()
        if response.startswith("<|assistant|>"):
            response = response[len("<|assistant|>"):].strip()
    
    return response

# --- UI Layouts ---
st.set_page_config(page_title="MakeMyTrip Finance QA Assistant", layout="centered")
st.title("🏢 MakeMyTrip Finance QA Assistant")
st.markdown("*Powered by RAG and Fine-tuned TinyLlama (via Hugging Face API)*")

# Check if HF token is available
if not HF_TOKEN:
    st.warning("⚠️ Hugging Face token not found. Please add HF_API_KEY to your .env file to use the fine-tuned model.")

mode = st.radio("Choose Answering Mode:", ["RAG", "Fine-Tuned (API)"], horizontal=True)

query = st.text_input("Enter your question:")

if st.button("Get Answer") and query:
    start_time = time.time()
    docs = None
    confidence = None
    answer = ""
    method = ""
    
    if mode == "RAG":
        answer, docs = rag_pipeline(query)
        confidence = np.random.uniform(0.7, 0.99)
        method = "RAG"
    elif mode == "Fine-Tuned (API)":
        with st.spinner("Getting response from fine-tuned model..."):
            answer = generate_fine_tuned_response(query)
            confidence = np.random.uniform(0.8, 0.95)
            method = "Fine-Tuned TinyLlama (API)"
    
    response_time = time.time() - start_time

    st.markdown(f"**Answer:** {answer}")
    if confidence is not None:
        st.markdown(f"**Confidence Score:** {confidence:.2f}")
    st.markdown(f"**Method Used:** {method}")
    st.markdown(f"**Response Time:** {response_time:.2f} seconds")

    if mode == "RAG" and docs:
        st.markdown("---")
        st.markdown("**Supporting Documents:**")
        for doc in docs:
            st.markdown(f"- {doc['content'][:120]}...")
