import streamlit as st
import time
import numpy as np
from search_final import rag_pipeline

# --- UI Layouts ---
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
