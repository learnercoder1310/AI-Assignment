import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract text
def extract_text_from_pdf(file):
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text("text") + "\n"
    return text

# Split into chunks
def chunk_text(text, chunk_size=200):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

# Process uploaded PDFs
def process_pdfs(files, index):
    all_chunks = []
    for f in files:
        text = extract_text_from_pdf(f)
        chunks = list(chunk_text(text))
        embeddings = model.encode(chunks)
        index.add(embeddings)
        all_chunks.extend(chunks)
    return index, all_chunks

