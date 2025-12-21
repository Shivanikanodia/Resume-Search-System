### AI Agent (RAG + LLM) - Generates Summaries of 100+ Resume in less then 20 seconds. 

An end-to-end pipeline that uses Retrieval-Augmented Generation (RAG), Sentence Transformers, Re-ranking and an LLM to generate concise, role aware candidate resume summaries.

## PROBLEM:

Recruiters and hiring managers often review hundreds of resumes manually which is tedious task. This agent extracts the information that actually matters to hiring teams and returns information based on query. 

## GOAL:

Serve Hiring Managers and Recruiters across functions (Engineering, Global Functions, Professional Services, etc.).
Let users query large resume sets by skills, experience, and team fit.
Reduce time-to-screen by surfacing the most relevant candidates and summaries.

---

### How It Works (Architecture)

1. **Data Loading:** Read resumes stored in Unity Catalog Volumes. (Can be integrated with ATS like greenhouse, workday or Oracle Taleo to fetch resumes in real time)

2. **Chunking & Embeddings:** Split resumes into ~800-character chunks, 

3. **Embedding with all-MiniLM-L6-v2 (384-dim) via SentenceTransformer.** Uses high dimentionality vectors, ensures relevant context and information. 

4. **Semantic Retrieval (Recall):** Cosine similarity over embeddings to fetch top-k relevant chunks. 

5. **Re-ranking (Precision):** Cross-encoder scores (query, chunk) pairs and reorders the retrieved set to keep the most relevant passages.

6. **Prompt Construction:** Insert top chunks into a concise, instruction-driven prompt with rules/constraints (focusing on role fit, impact, skills (Hiring Manger Expectations).

7. **LLM Generation:** Call Databricks-hosted Llama endpoint to produce the final summary and answers. 

8. **Evaluation & Observability:** Track latency, error rate and  retrieval quality. 
 
---

#### Project Structure:

1. Read & Chunking: read_chunk_resume() splits text → list of chunks.

2. Embeddings: SentenceTransformer("all-MiniLM-L6-v2").encode(chunks) → vectors.

3. Retrieve: cosine similarity to get top-k chunks.

4. Re-rank: using cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2) for re-scores.

5. Prompt: build_prompt(context, query) adds rules/instructions + top chunks.

6. Generate: call Databricks Llama endpoint with max_tokens limit.

7. Log: latency, errors, rank metrics, token usage.

   
---

**Model Output** and **Evaluation Metrics:**::

<img width="852" height="130" alt="Screenshot 2025-11-06 at 16 00 45" src="https://github.com/user-attachments/assets/c6c54f71-4c18-4c74-9086-f6d29a68dccf" />

<img width="985" height="289" alt="Screenshot 2025-11-06 at 16 00 39" src="https://github.com/user-attachments/assets/4ed11558-2f28-4749-ac75-071ddf59fe08" />

The Output and revelancy can be improved using meta-data filtering, BM25 and Indexing.

---

**Instructions to Run:**

**Backend (Databricks + Python)** 

Create a Databricks token and Llama endpoint (model serving).

**Install deps:**

pip install sentence-transformers torch requests faiss-cpu

**Configure environment variables:**

DATABRICKS_HOST, DATABRICKS_TOKEN, LLM_ENDPOINT (your model serving URL)

Load resumes, Change path your specified path location, run embedding + index build, then start the API script that calls the Llama endpoint

---

📜 **License**

This project is intended for educational and internal enterprise use.
