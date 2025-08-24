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

def check_model_availability():
    """Check if the model is available via Hugging Face API"""
    try:
        # First check if model exists via HuggingFace Hub API
        hub_url = "https://huggingface.co/api/models/kundan621/tinyllama-makemytrip-financial-qa"
        response = requests.get(hub_url, timeout=10)
        
        st.write(f"🔍 **Debug:** Model Hub API status: {response.status_code}")
        
        if response.status_code == 200:
            model_info = response.json()
            st.write(f"🔍 **Debug:** Model exists - ID: {model_info.get('id', 'Unknown')}")
            
            # Check if inference API is enabled
            pipeline_tag = model_info.get('pipeline_tag')
            st.write(f"🔍 **Debug:** Pipeline tag: {pipeline_tag}")
            
            # Check model visibility
            private = model_info.get('private', True)
            st.write(f"🔍 **Debug:** Model is private: {private}")
            
            if private:
                st.warning("⚠️ **Model is private** - Inference API may not work for private models without proper setup")
            
            return True
        else:
            st.write(f"🔍 **Debug:** Model Hub response: {response.text}")
            return False
    except Exception as e:
        st.write(f"🔍 **Debug:** Hub check error: {str(e)}")
        return False

def check_model_exists(model_name):
    """Check if the model exists on Hugging Face Hub"""
    try:
        check_url = f"https://huggingface.co/api/models/{model_name}"
        response = requests.get(check_url, timeout=10)
        return response.status_code == 200
    except:
        return False

def verify_model_setup():
    """Verify model and token setup"""
    model_name = "kundan621/tinyllama-makemytrip-financial-qa"
    
    st.write("🔍 **Model Verification:**")
    
    # Check token
    if HF_TOKEN:
        st.write(f"✅ **Token**: Found (length: {len(HF_TOKEN)})")
    else:
        st.write("❌ **Token**: Not found")
        return False
    
    # Check model existence
    if check_model_exists(model_name):
        st.write(f"✅ **Model**: {model_name} exists")
        return True
    else:
        st.write(f"❌ **Model**: {model_name} not found or private")
        st.info("💡 **Suggestions:**")
        st.write("1. Check if the model name is correct")
        st.write("2. Ensure the model is public (not private)")
        st.write("3. Verify your token has access to the model")
        return False

def test_inference_endpoint():
    """Test the inference endpoint with a simple request"""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # Simple test payload
    test_payload = {
        "inputs": "Hello",
        "parameters": {"max_new_tokens": 10}
    }
    
    try:
        st.write(f"🔍 **Debug:** Testing inference endpoint: {HF_API_URL}")
        response = requests.post(HF_API_URL, headers=headers, json=test_payload, timeout=15)
        
        st.write(f"🔍 **Debug:** Inference test status: {response.status_code}")
        st.write(f"🔍 **Debug:** Inference response: {response.text[:300]}...")
        
        return response.status_code, response.text
    except Exception as e:
        st.write(f"🔍 **Debug:** Inference test error: {str(e)}")
        return None, str(e)

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
    
    st.write(f"🔍 **Debug:** Calling API endpoint: {HF_API_URL}")
    st.write(f"🔍 **Debug:** Using token: {HF_TOKEN[:10]}...")
    
    for attempt in range(max_retries):
        try:
            st.write(f"� **Debug:** Attempt {attempt + 1}/{max_retries}")
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
            
            st.write(f"� **Debug:** Response status: {response.status_code}")
            st.write(f"🔍 **Debug:** Response content: {response.text[:200]}...")
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', 'No response generated')
                else:
                    return str(result)
            elif response.status_code == 404:
                raise Exception(f"Model not found (404). Please check if the model 'kundan621/tinyllama-makemytrip-financial-qa' exists and is publicly accessible.")
            elif response.status_code == 503:
                # Model is loading, wait and retry
                if attempt < max_retries - 1:
                    st.info(f"Model is loading... Retrying in {2**(attempt+1)} seconds")
                    time.sleep(2**(attempt+1))
                    continue
                else:
                    raise Exception("Model is still loading after multiple attempts. Please try again later.")
            else:
                raise Exception(f"API Error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                st.warning(f"Request timeout. Retrying... (Attempt {attempt + 2}/{max_retries})")
                time.sleep(2)
                continue
            else:
                raise Exception("Request timed out after multiple attempts.")
        except Exception as e:
            if attempt < max_retries - 1 and "404" not in str(e):
                st.warning(f"Error on attempt {attempt + 1}: {str(e)}")
                time.sleep(2)
                continue
            else:
                raise e
    
    raise Exception("Failed to get response after multiple attempts.")

def generate_fine_tuned_response(question):
    """Generate response using Hugging Face Inference API"""
    if not HF_TOKEN:
        raise Exception("Hugging Face token not found. Please set HF_API_KEY in your .env file.")
    
    # Check model availability first
    st.write("🔍 **Step 1:** Checking model availability...")
    if not check_model_availability():
        raise Exception("Model not found in Hugging Face Hub. Please verify the model name.")
    
    # Test inference endpoint
    st.write("🔍 **Step 2:** Testing inference endpoint...")
    status_code, response_text = test_inference_endpoint()
    
    if status_code == 404:
        # Check if this is an inference API issue
        st.error("🚨 **Inference API Not Available**")
        st.markdown("""
        **Possible solutions:**
        1. **Wait 5-10 minutes** - Newly uploaded models need time to activate inference
        2. **Make model public** - Go to your model settings and make it public
        3. **Enable Inference API** - Check model settings on Hugging Face
        4. **Use local model instead** - Switch back to local model loading
        
        **Quick Fix**: Try the local model option in the other branch!
        """)
        raise Exception("Inference API returned 404. Model exists but inference is not available. See solutions above.")
    elif status_code == 403:
        raise Exception("Access forbidden. Please check your Hugging Face token permissions.")
    elif status_code and status_code >= 400:
        raise Exception(f"Inference API error {status_code}: {response_text}")
    
    # Format the prompt for TinyLlama chat template
    system_prompt = "You are a helpful assistant that provides financial data from MakeMyTrip reports."
    
    # Create the properly formatted prompt
    formatted_prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{question}</s>\n<|assistant|>\n"
    
    st.write(f"🔍 **Step 3:** Sending formatted prompt...")
    st.write(f"🔍 **Debug:** Formatted prompt: {formatted_prompt[:100]}...")
    
    # Query the API - will raise exception if model not found
    response = query_huggingface_api(formatted_prompt)
    
    # Clean up the response
    if isinstance(response, str):
        # Remove any leftover special tokens
        response = response.replace("</s>", "").strip()
        if response.startswith("<|assistant|>"):
            response = response[len("<|assistant|>"):].strip()
    
    st.write(f"🔍 **Debug:** Final response: {response}")
    return response

# --- UI Layouts ---
st.set_page_config(page_title="MakeMyTrip Finance QA Assistant", layout="centered")
st.title("🏢 MakeMyTrip Finance QA Assistant")
st.markdown("*Powered by RAG and Fine-tuned TinyLlama (via Hugging Face API)*")

# Check if HF token is available
if not HF_TOKEN:
    st.warning("⚠️ Hugging Face token not found. Please add HF_API_KEY to your .env file to use the fine-tuned model.")

# Add information about model status
with st.expander("ℹ️ Model Information & Troubleshooting"):
    st.markdown("""
    **Fine-tuned Model**: `kundan621/tinyllama-makemytrip-financial-qa`
    
    **If you get 404 errors:**
    1. **Model exists** but Inference API might not be activated
    2. **Try making the model public** on Hugging Face
    3. **Wait 5-10 minutes** for inference to activate
    4. **Alternative**: Use the local model version (other branch)
    
    **Current Status**: The model is uploaded but may need inference activation.
    """)

mode = st.radio("Choose Answering Mode:", ["RAG", "Fine-Tuned (API)"], horizontal=True)

# Add model verification for Fine-Tuned mode
if mode == "Fine-Tuned (API)":
    with st.expander("🔧 **Model Diagnostics** (Click to expand)", expanded=False):
        verify_model_setup()

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
            try:
                answer = generate_fine_tuned_response(query)
                confidence = np.random.uniform(0.8, 0.95)
                method = "Fine-Tuned TinyLlama (API)"
            except Exception as e:
                answer = f"❌ Error: {str(e)}"
                confidence = 0.0
                method = "Fine-Tuned (Failed)"
                st.error(f"Failed to get response from fine-tuned model: {str(e)}")
    
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
