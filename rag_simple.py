import os, glob, re
from typing import List, Tuple
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

def read_docs(doc_dir: str) -> List[Tuple[str, str]]:
    paths = glob.glob(os.path.join(doc_dir, "*"))
    docs = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                docs.append((os.path.basename(p), f.read()))
        except Exception:
            pass
    return docs

def split_into_chunks(text: str, chunk_size: int = 500):
    # naive chunker by sentences
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, buf = [], ""
    for s in sents:
        if len(buf) + len(s) <= chunk_size:
            buf += (" " if buf else "") + s
        else:
            if buf: chunks.append(buf)
            buf = s
    if buf: chunks.append(buf)
    return chunks

def build_tfidf_index(chunks: List[str]):
    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform(chunks)
    return vec, mat

def retrieve(query: str, vec, mat, chunks: List[str], top_k: int = 3):
    qv = vec.transform([query])
    sims = cosine_similarity(qv, mat).ravel()
    idxs = sims.argsort()[::-1][:top_k]
    return [(chunks[i], float(sims[i])) for i in idxs]

