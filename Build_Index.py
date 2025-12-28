import glob
import pickle
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from preprocessing import read_and_chunk_resume, anonymize, extract_metadata

def build_index(resume_paths, out_dir="resume_index"):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = []
    chunk_meta = {}
    resume_meta = {}

    for rp in resume_paths:
        print(f"Processing: {rp}")  # DEBUG LINE

        chunks = read_and_chunk_resume(rp)
        anon_chunks = [anonymize(c) for c in chunks]

        # ✅ THIS LINE IS CRITICAL
        resume_meta[rp] = extract_metadata(" ".join(chunks))

        embs = model.encode(anon_chunks, normalize_embeddings=True)

        for e, c in zip(embs, anon_chunks):
            idx = len(embeddings)
            embeddings.append(e)
            chunk_meta[idx] = {
                "resume_path": rp,
                "text": c
            }

    if not embeddings:
        raise ValueError("No embeddings created — check resume ingestion.")

    index = faiss.IndexFlatIP(len(embeddings[0]))
    index.add(np.array(embeddings).astype("float32"))

    Path(out_dir).mkdir(exist_ok=True)

    faiss.write_index(index, f"{out_dir}/faiss.index")

    with open(f"{out_dir}/chunk_meta.pkl", "wb") as f:
        pickle.dump(chunk_meta, f)

    # ✅ THIS IS WHAT YOU ARE MISSING
    with open(f"{out_dir}/resume_meta.pkl", "wb") as f:
        pickle.dump(resume_meta, f)

    print("✅ Index + metadata built successfully")

if __name__ == "__main__":
    resume_paths = glob.glob("data/resumes/*.txt")
    print("Found resumes:", resume_paths)

    if not resume_paths:
        raise ValueError("❌ No resumes found")

    build_index(resume_paths)
