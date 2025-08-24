from sentence_transformers import SentenceTransformer
import openai

model = SentenceTransformer("all-MiniLM-L6-v2")

def query_answer(question, index, doc_chunks):
    q_emb = model.encode([question])
    D, I = index.search(q_emb, k=3)  # top 3 results
    retrieved = [doc_chunks[i] for i in I[0]]

    # Create GPT prompt
    context = "\n".join(retrieved)
    prompt = f"Answer the question using only this context:\n\n{context}\n\nQuestion: {question}\nAnswer:"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"], retrieved

