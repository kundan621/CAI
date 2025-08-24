
import os
from dotenv import load_dotenv
import re
import pickle
import faiss
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
import requests
import json
from openai import OpenAI
import logging

load_dotenv()
# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ...rest of your imports...

# ---------------- Paths & Models ----------------
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUT_DIR       = "data/index_merged"

FAISS_PATH = os.path.join(OUT_DIR, "faiss_merged.index")
BM25_PATH  = os.path.join(OUT_DIR, "bm25_merged.pkl")
META_PATH  = os.path.join(OUT_DIR, "meta_merged.pkl")

BLOCKED_TERMS = ["weather","cricket","movie","song","football","holiday",
                 "travel","recipe","music","game","sports","politics","election"]

FINANCE_DOMAINS = [
    "financial reporting","balance sheet","income statement","assets and liabilities",
    "equity","revenue","profit and loss","goodwill impairment","cash flow","dividends",
    "taxation","investment","valuation","capital structure","ownership interests",
    "subsidiaries","shareholders equity","expenses","earnings","debt","amortization","depreciation"
]

ALLOWED_COMPANY = ["make my trip","mmt"]

# crude regex to detect "company-like" words (any capitalized word(s) followed by Ltd, Inc, Company, etc.)
COMPANY_PATTERN = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Ltd|Limited|Inc|Corporation|Corp|LLC|Group|Company|Bank))\b", re.IGNORECASE)

# ---------------- Load Indexes ----------------
logger.info("Loading FAISS, BM25, metadata, and models...")
try:
    faiss_index = faiss.read_index(FAISS_PATH)
    with open(BM25_PATH, "rb") as f:
        bm25_obj = pickle.load(f)
    bm25 = bm25_obj["bm25"]
    with open(META_PATH, "rb") as f:
        meta: List[Dict] = pickle.load(f)
    embed_model = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(CROSS_ENCODER)
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        logger.error("HF_API_KEY environment variable not set. Please check your .env file or environment.")
        raise ValueError("HF_API_KEY environment variable not set.")
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=api_key
    )
except Exception as e:
    logger.error(f"Error loading models or indexes: {e}")
    raise

# ---------------- Hugging Face Mistral API ----------------
#HF_TOKEN = "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"
#HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai"

def get_mistral_answer(query: str, context: str) -> str:
    """
    Calls Mistral 7B Instruct API via Hugging Face Inference API.
    Adds error handling and logging.
    """
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer in full sentences using context."
    try:
        logger.info(f"Calling Mistral API for query: {query}")
        completion = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        answer = str(completion.choices[0].message.content)
        logger.info(f"Mistral API response: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Error in Mistral API call: {e}")
        return f"Error fetching answer from LLM: {e}"

# ---------------- Guardrails ----------------
finance_embeds = embed_model.encode(FINANCE_DOMAINS, convert_to_tensor=True)

def validate_query(query: str, threshold: float = 0.5) -> bool:
    q_lower = query.lower()
    
    # Blocklist check
    if any(bad in q_lower for bad in BLOCKED_TERMS):
        print("[Guardrail] Rejected by blocklist.")
        return False

    # Check for company mentions
    companies_found = COMPANY_PATTERN.findall(query)
    if companies_found:
        # If any company is mentioned, only allow MakeMyTrip
        if not any(ALLOWED_COMPANY in c.lower() for c in companies_found):
            print(f"[Guardrail] Rejected: company mention {companies_found}, not {ALLOWED_COMPANY}.")
            return False
    
    # Semantic similarity check with financial domain
    q_emb = embed_model.encode(query, convert_to_tensor=True)
    sim_scores = util.cos_sim(q_emb, finance_embeds)
    max_score = float(sim_scores.max())

    if max_score > threshold:
        print(f"[Guardrail] Accepted (semantic match {max_score:.2f})")
        return True
    else:
        print(f"[Guardrail] Rejected (low semantic score {max_score:.2f})")
        return False

#-------------------Output Guardrail------------------
def validate_output(answer: str, context_docs: List[Dict]) -> str:
    combined_context = " ".join([doc["content"].lower() for doc in context_docs])
    if answer.lower() in combined_context:
        return answer
    return "The information could not be verified in the financial statement attached."

# ---------------- Preprocess ----------------
def preprocess_query(query: str, remove_stopwords: bool = True) -> str:
    query = query.lower()
    query = re.sub(r"[^a-z0-9\s]", " ", query)
    tokens = query.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

# ---------------- Hybrid Retrieval ----------------
def hybrid_candidates(query: str, candidate_k: int = 50, alpha: float = 0.5) -> List[int]:
    q_emb = embed_model.encode([preprocess_query(query, remove_stopwords=False)], convert_to_numpy=True, normalize_embeddings=True)
    faiss_scores, faiss_ids = faiss_index.search(q_emb, max(candidate_k, 50))
    faiss_ids = faiss_ids[0]
    faiss_scores = faiss_scores[0]

    tokenized_query = preprocess_query(query).split()
    bm25_scores = bm25.get_scores(tokenized_query)

    topN = max(candidate_k, 50)
    bm25_top = np.argsort(bm25_scores)[::-1][:topN]
    faiss_top = faiss_ids[:topN]
    union_ids = np.unique(np.concatenate([bm25_top, faiss_top]))

    faiss_score_map = {int(i): float(s) for i, s in zip(faiss_ids, faiss_scores)}
    f_arr = np.array([faiss_score_map.get(int(i), -1.0) for i in union_ids], dtype=float)
    f_min = np.min(f_arr)
    if np.any(f_arr < 0):
        f_arr = np.where(f_arr < 0, f_min, f_arr)
    b_arr = np.array([bm25_scores[int(i)] for i in union_ids], dtype=float)

    def _norm(x): return (x - np.min(x)) / (np.ptp(x) + 1e-9)
    combined = alpha * _norm(f_arr) + (1 - alpha) * _norm(b_arr)
    order = np.argsort(combined)[::-1]
    return union_ids[order][:candidate_k].tolist()

# ---------------- Cross-Encoder Rerank ----------------
def rerank_cross_encoder(query: str, cand_ids: List[int], top_k: int = 10) -> List[Dict]:
    pairs = [(query, meta[i]["content"]) for i in cand_ids]
    scores = reranker.predict(pairs)
    order = np.argsort(scores)[::-1][:top_k]
    return [{"id": cand_ids[i], "chunk_size": meta[cand_ids[i]]["chunk_size"], "content": meta[cand_ids[i]]["content"], "rerank_score": float(scores[i])} for i in order]

# ---------------- Extract Numeric ----------------
def extract_value_for_year_and_concept(year: str, concept: str, context_docs: List[Dict]) -> str:
    target_year = str(year)
    concept_lower = concept.lower()
    for doc in context_docs:
        text = doc.get("content", "")
        lines = [line for line in text.split("\n") if line.strip() and any(c.isdigit() for c in line)]
        header_idx = None
        year_to_col = {}
        for idx, line in enumerate(lines):
            years_in_line = re.findall(r"20\d{2}", line)
            if years_in_line:
                for col_idx, y in enumerate(years_in_line):
                    year_to_col[y] = col_idx
                header_idx = idx
                break
        if target_year not in year_to_col or header_idx is None:
            continue
        for line in lines[header_idx+1:]:
            if concept_lower in line.lower():
                cols = re.split(r"\s{2,}|\t", line)
                col_idx = year_to_col[target_year]
                if col_idx < len(cols):
                    return cols[col_idx].replace(",", "")
    return ""

# ---------------- RAG Pipeline ----------------
def rag_pipeline(query: str, top_k: int = 5, candidate_k: int = 50, alpha: float = 0.6):
    logger.info(f"Received query: {query}")
    try:
        if not validate_query(query):
            logger.warning("Query rejected: Not finance-related.")
            return "Query rejected: Please ask finance-related questions.", []

        cand_ids = hybrid_candidates(query, candidate_k=candidate_k, alpha=alpha)
        logger.info(f"Hybrid candidates retrieved: {cand_ids}")
        reranked = rerank_cross_encoder(query, cand_ids, top_k=top_k)
        logger.info(f"Reranked top docs: {[d['id'] for d in reranked]}")

        year_match = re.search(r"(20\d{2})", query)
        year = year_match.group(0) if year_match else None
        concept = re.sub(r"for the year 20\d{2}", "", query, flags=re.IGNORECASE).strip()

        year_specific_answer = None
        if year and concept:
            year_specific_answer = extract_value_for_year_and_concept(year, concept, reranked)
            logger.info(f"Year-specific answer: {year_specific_answer}")

        if year_specific_answer:
            answer = year_specific_answer
        else:
            # Pass top 5 chunks as context
            context_text = "\n".join([d["content"] for d in reranked])
            answer = get_mistral_answer(query, context_text)
        final_answer = answer #validate_output(answer, reranked)
        logger.info(f"Final Answer: {final_answer}")
        return final_answer, reranked
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {e}")
        return f"Error in RAG pipeline: {e}", []

# ---------------- Example ----------------
if __name__ == "__main__":
    query = "What is the Balance as at March 31, 2024 for accumulated deficit?"
    answer, top_docs = rag_pipeline(query)

    print(f"\nQuery: {query}")
    print("\nFinal Answer:\n", answer)
    print("\nTop supporting docs:")
    for doc in top_docs:
        print(f"[{doc['id']}] (chunk={doc['chunk_size']}, score={doc['rerank_score']:.3f}) -> {doc['content'][:120]}...")
