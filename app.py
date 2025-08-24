import streamlit as st
import fitz  # PyMuPDF
import faiss
from sentence_transformers import SentenceTransformer
import openai
import os

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS index
dimension = 384  # embedding size for MiniLM
index = faiss.IndexFlatL2(dimension)

# Store chunks
doc_chunks = []

# PDF text extractor
def extract_text_from_pdf(file):
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text("text") + "\n"
    return text

# Chunking text
def chunk_text(text, chunk_size=200):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

# Add embeddings to FAISS
def add_to_index(chunks):
    embeddings = model.encode(chunks)
    index.add(embeddings)
    doc_chunks.extend(chunks)

# Query function
def query_answer(question):
    q_emb = model.encode([question])
    D, I = index.search(q_emb, k=3)  # top 3 results
    retrieved = [doc_chunks[i] for i in I[0]]

    # Build context for GPT
    context = "\n".join(retrieved)
    prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"], retrieved


# Streamlit UI
st.title("📄 AI PDF Assistant")

uploaded_files = st.file_uploader("Upload PDF(s)", accept_multiple_files=True, type="pdf")

if uploaded_files:
    for f in uploaded_files:
        text = extract_text_from_pdf(f)
        chunks = list(chunk_text(text))
        add_to_index(chunks)
    st.success("✅ PDF(s) processed!")

query = st.text_input("Ask a question:")
if query:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ Please set your OpenAI API Key as environment variable: `export OPENAI_API_KEY=your_key`")
    else:
        answer, refs = query_answer(query)
        st.write("### ✅ Answer:", answer)
        st.write("#### 📌 Context:", refs)

