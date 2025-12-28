import streamlit as st
import pickle
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path

INDEX_DIR = "resume_index"

# ---------- Prompt Engineering ----------
def format_prompt(context, query):
    return f"""
You are helping a recruiter evaluate a resume.

STRICT RULES:
- Bullet points only
- Max 5 bullets
- Each bullet ≤ 15 words
- No repetition
- No paragraphs

Resume content:
{context}

Hiring query:
{query}

Output format:
• Relevant experience
• Skills match
• Tools & technologies
• Strengths
• Gaps
"""

# ---------- LLM Call (Ollama) ----------
def call_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# ---------- Load Models & Index ----------
@st.cache_resource
def load_resources():
    index = faiss.read_index(f"{INDEX_DIR}/faiss.index")
    chunk_meta = pickle.load(open(f"{INDEX_DIR}/chunk_meta.pkl", "rb"))
    resume_meta = pickle.load(open(f"{INDEX_DIR}/resume_meta.pkl", "rb"))

    bi = SentenceTransformer("all-MiniLM-L6-v2")
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return index, chunk_meta, resume_meta, bi, ce

index, chunk_meta, resume_meta, bi, ce = load_resources()

# ---------- Retrieval ----------
def retrieve_candidates(query, top_k=5):
    q_emb = bi.encode([query], normalize_embeddings=True).astype("float32")

    k = min(50, index.ntotal)
    scores, ids = index.search(q_emb, k)

    resume_scores = {}
    resume_chunks = {}

    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue

        rp = chunk_meta[idx]["resume_path"]
        resume_scores[rp] = max(resume_scores.get(rp, 0), score)
        resume_chunks.setdefault(rp, []).append(chunk_meta[idx]["text"])

    ranked = sorted(resume_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k], resume_chunks

# ---------- Reranking ----------
def rerank(query, ranked, resume_chunks):
    pairs = [(query, "\n".join(resume_chunks[rp][:5])) for rp, _ in ranked]
    scores = ce.predict(pairs)
    return sorted(zip(ranked, scores), key=lambda x: x[1], reverse=True)

# ---------- Streamlit UI ----------
st.title("🔍 Local AI Resume Search")

query = st.text_area("Enter hiring query")

if st.button("Search") and query.strip():

    ranked, chunks = retrieve_candidates(query)
    reranked = rerank(query, ranked, chunks)

    for ((rp, bi_score), ce_score) in reranked:
        cosine_pct = (bi_score + 1) / 2 * 100

        with st.expander(f"📄 {Path(rp).name}"):
            st.metric("Semantic Match (FAISS)", f"{cosine_pct:.0f}%")
            st.metric("Final Relevance (Cross-Encoder)", f"{ce_score:.2f}")

            context = "\n".join(chunks[rp][:3])
            prompt = format_prompt(context, query)
            summary = call_llm(prompt)

            st.markdown("**Resume Summary**")
            st.markdown(summary)
