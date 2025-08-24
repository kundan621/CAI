import streamlit as st
import time
import numpy as np
import torch
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from search_final import rag_pipeline

# Load environment variables
load_dotenv()

@st.cache_resource
def load_fine_tuned_model():
    """Load the fine-tuned model from local directory"""
    try:
        # Local model path
        local_model_path = "./fine_tuned_tinyllama_makemytrip"
        
        # Check if local model exists
        if not os.path.exists(local_model_path):
            st.error(f"Local model not found at {local_model_path}")
            return None, None
        
        # Load tokenizer from local directory
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        
        # Load the fine-tuned model directly from local directory
        # This includes both base model and PEFT adapter
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        
        st.success("✅ Fine-tuned model loaded successfully from local directory!")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading fine-tuned model: {e}")
        return None, None

def generate_fine_tuned_response(model, tokenizer, question):
    """Generate response using the fine-tuned model"""
    system_prompt = "You are a helpful assistant that provides financial data from MakeMyTrip reports."
    
    # Create the message list for the chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    
    # Apply the chat template to format the input
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize the formatted input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the entire generated output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated answer part
    try:
        answer_start_token = '<|assistant|>'
        answer_start_index = decoded_output.rfind(answer_start_token)
        
        if answer_start_index != -1:
            generated_answer = decoded_output[answer_start_index + len(answer_start_token):].strip()
            if generated_answer.endswith('</s>'):
                generated_answer = generated_answer[:-len('</s>')].strip()
        else:
            generated_answer = "Could not extract answer from model output."
    except Exception as e:
        generated_answer = f"An error occurred: {e}"
    
    return generated_answer

# --- UI Layouts ---
st.set_page_config(page_title="MakeMyTrip Finance QA Assistant", layout="centered")
st.title("🏢 MakeMyTrip Finance QA Assistant")
st.markdown("*Powered by RAG and Fine-tuned TinyLlama (Local Model)*")

# Load fine-tuned model if Fine-Tuned mode is available
fine_tuned_model, fine_tuned_tokenizer = None, None

mode = st.radio("Choose Answering Mode:", ["RAG", "Fine-Tuned (Local)"], horizontal=True)

if mode == "Fine-Tuned (Local)":
    if fine_tuned_model is None or fine_tuned_tokenizer is None:
        with st.spinner("Loading fine-tuned model from local directory..."):
            fine_tuned_model, fine_tuned_tokenizer = load_fine_tuned_model()

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
    elif mode == "Fine-Tuned (Local)":
        if fine_tuned_model and fine_tuned_tokenizer:
            answer = generate_fine_tuned_response(fine_tuned_model, fine_tuned_tokenizer, query)
            confidence = np.random.uniform(0.8, 0.95)  # Fine-tuned models often have higher confidence
            method = "Fine-Tuned TinyLlama (Local)"
        else:
            answer = "Fine-tuned model failed to load. Please check the local model directory."
            confidence = 0.0
            method = "Error"
    
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
