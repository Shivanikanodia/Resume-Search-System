### AI POWERED RESUME RANKING SYSTEM: 

An end-to-end semantic resume search and ranking system that matches resumes to hiring queries using vector search, cross-encoder reranking, and optional LLM summarization.

This project mirrors production-grade hiring intelligence systems by cleanly separating offline indexing from online inference.

## PROBLEM:

- Recruiters often screen hundreds of resumes manually using keyword search, which is slow, inconsistent, biased toward wording rather than meaning
- This project solves that by: understanding semantic similarity, ranking candidates, not keywords and providing concise AI-generated summaries for decision support.

## GOAL:

Serve Hiring Managers and Recruiters across functions (Engineering, Global Functions, Professional Services, etc.).
Let users query large resume sets by skills, experience, and team fit.
Reduce time-to-screen by surfacing the most relevant candidates and summaries.

---

### How It Works (Architecture)

Offline (Run Once)
Resumes
→ Read & Chunk
→ Anonymize PII
→ Generate Embeddings
→ Build FAISS Index
→ Store Metadata
---
Online (Every Query)
User Query
→ Vector Search (FAISS)
→ Resume-level Aggregation
→ Cross-Encoder Re-Ranking
→ (Optional) LLM Summarization
→ Streamlit UI
---

#### Key Design Principles

1️⃣ Offline vs Online Separation:

Index construction is compute-heavy and runs once

Query-time inference is fast and lightweight

Prevents recomputation and improves scalability

2️⃣ Deterministic Ranking, Generative Summaries:

Ranking is driven by retrieval + reranking

LLMs are used only for post-hoc summarization

Avoids hallucinations and keeps decisions explainable

3️⃣ Privacy-First Design:

Names, emails, and phone numbers are removed before embedding

Prevents sensitive data from entering vector space
   
---

Resume_Ranking_System/
│
├── data/
│   └── resumes/              # Input resumes (.txt, extendable to PDF/DOCX)
│
├── resume_index/             # Generated offline
│   ├── faiss.index
│   ├── chunk_meta.pkl
│   └── resume_meta.pkl
│
├── preprocessing.py          # Resume reading, chunking, anonymization
├── build_index.py            # Offline FAISS index builder
├── app.py                    # Streamlit app (online inference)
└── README.md


**Model Output** and **Evaluation Metrics:**::

<img width="852" height="130" alt="Screenshot 2025-11-06 at 16 00 45" src="https://github.com/user-attachments/assets/c6c54f71-4c18-4c74-9086-f6d29a68dccf" />

---

## Technology Stack

Python 3.10
FAISS – fast vector similarity search
Sentence-Transformers – semantic embeddings
Cross-Encoder (MiniLM) – reranking for precision
Streamlit – interactive UI
Ollama (Llama 3) – optional local LLM for summarization

---

## Setup Instructions (Local)
1️⃣ Create Environment
conda create -n resume_rank python=3.10
conda activate resume_rank

2️⃣ Install Dependencies
conda install -c pytorch faiss-cpu
python -m pip install sentence-transformers streamlit requests numpy

###  Step 1: Build the Index (Offline)

Place resumes inside:
data/resumes/

Run:
python build_index.py

This generates:

resume_index/
├── faiss.index
├── chunk_meta.pkl
└── resume_meta.pkl

This step is executed only when resumes change.

### Step 2: Run the Application (Online)
Start the local LLM (optional but recommended)
ollama pull llama3
ollama serve

Launch the Streamlit app
python -m streamlit run app.py

### Open the browser and enter queries like:

Data analyst with Python, SQL, and Tableau experience

----

### What Happens During a Search:

- Query Embedding
- Query is converted to a vector
- FAISS Vector Search
- Retrieves semantically similar resume chunks
- Resume-Level Aggregation
- Prevents longer resumes from dominating
- Cross-Encoder Re-Ranking
- Improves final ordering accuracy
- Optional LLM Summarization
- Generates structured, concise summaries
- Uses only top-ranked content
- Transparent relevance scores

## Scoring Explained:

Semantic Match (FAISS)
Cosine similarity 
Final Relevance (Cross-Encoder)
Stronger signal used for final ordering.

Scores are relative, not absolute — ranking quality matters more than raw values.

## Business Impact:
1. Faster Hiring

Reduces resume screening time from hours to seconds

2. Better Candidate Quality:

Finds strong matches beyond keyword overlap

3. Reduced Bias

Anonymization removes identity-based signals

4. Scalable

Works for tens or thousands of resumes

5. Trustworthy AI

LLM does not make decisions
Used only for summarization and explanation

---

📜 **License**

This project is intended for educational and internal enterprise use.
