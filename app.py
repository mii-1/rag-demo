import os
import streamlit as st
from dotenv import load_dotenv
from rag_simple import read_docs, split_into_chunks, build_tfidf_index, retrieve

def get_llm_client():
    from openai import OpenAI, AzureOpenAI
    if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-06-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        ), os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-4o-mini"
    elif os.getenv("OPENAI_API_KEY"):
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY")), "gpt-4o-mini"
    return None, None

def llm_summarize(client, model_name, question, contexts):
    ctx_text = "\n\n".join([f"[{i+1}] {c[0][:500]}" for i, c in enumerate(contexts)])
    prompt = f"Use the CONTEXT below to answer. Cite sources [1], [2], ... if used.\n\nCONTEXT:\n{ctx_text}\n\nQUESTION: {question}\n"
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role":"system","content":"You are a concise enterprise assistant."},
                  {"role":"user","content":prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()

load_dotenv()
st.title("🔎 Minimal RAG Demo")

docs = read_docs("docs")
if not docs:
    st.warning("No docs found in ./docs")
else:
    # build chunks
    chunks = []
    for name, text in docs:
        for c in split_into_chunks(text, 400):
            chunks.append(f"{name}: {c}")

    vec, mat = build_tfidf_index(chunks)

q = st.text_input("Ask a question about the documents")
if st.button("Retrieve & Answer") and q:
    top = retrieve(q, vec, mat, chunks, top_k=3)
    st.subheader("Top Passages")
    for i, (passage, score) in enumerate(top, 1):
        st.write(f"[{i}] (score={score:.3f}) {passage}")

    client, model = get_llm_client()
    if client:
        try:
            ans = llm_summarize(client, model, q, top)
            st.success(ans)
        except Exception as e:
            st.error(f"LLM failed: {e}")
            st.info("You can still read the top passages above.")
    else:
        st.info("No LLM key found; showing retrieved passages only.")

