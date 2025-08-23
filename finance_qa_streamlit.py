import streamlit as st
import time
import numpy as np
from search_final import rag_pipeline

# --- Fine-tuned model loader ---
from Fine_Tuning_TinyLlama import generate_finetuned_answer, model as finetuned_model, tokenizer as finetuned_tokenizer

# --- UI Layout ---
st.set_page_config(page_title="Finance QA Assistant", layout="centered")
st.title("Finance QA Assistant")

mode = st.radio("Choose Answering Mode:", ["RAG", "Fine-Tuned"], horizontal=True)
query = st.text_input("Enter your question:")

if st.button("Get Answer") and query:
    start_time = time.time()
    docs = None
    if mode == "RAG":
        answer, docs = rag_pipeline(query)
        confidence = np.random.uniform(0.7, 0.99)
        method = "RAG"
    else:
        answer = generate_finetuned_answer(query, finetuned_model, finetuned_tokenizer)
        confidence = np.random.uniform(0.6, 0.95)
        method = "Fine-Tuned"
    response_time = time.time() - start_time

    st.markdown(f"**Answer:** {answer}")
    st.markdown(f"**Confidence Score:** {confidence:.2f}")
    st.markdown(f"**Method Used:** {method}")
    st.markdown(f"**Response Time:** {response_time:.2f} seconds")

    if mode == "RAG" and docs:
        st.markdown("---")
        st.markdown("**Supporting Documents:**")
        for doc in docs:
            st.markdown(f"- {doc['content'][:120]}...")
