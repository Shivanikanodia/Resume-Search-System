import re
import textwrap
from pypdf import PdfReader
from docx import Document

# -------- Resume Reading --------
def read_resume(path):
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    elif path.endswith(".pdf"):
        reader = PdfReader(path)
        return " ".join(page.extract_text() or "" for page in reader.pages)

    elif path.endswith(".docx"):
        doc = Document(path)
        return " ".join(p.text for p in doc.paragraphs)

    else:
        raise ValueError(f"Unsupported file type: {path}")

# -------- Chunking --------
def read_and_chunk_resume(path, width=900):
    text = read_resume(path)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return textwrap.wrap(text, width)

# -------- Anonymization --------
def anonymize(text):
    text = re.sub(r"\S+@\S+", "[EMAIL]", text)
    text = re.sub(r"\b\d{10}\b", "[PHONE]", text)
    text = re.sub(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "[CANDIDATE]", text)
    return text

# -------- Metadata Extraction --------
def extract_metadata(text):
    skills = ["python", "sql", "tableau", "databricks"]
    low = text.lower()
    return {
        "skills": [s for s in skills if s in low]
    }
