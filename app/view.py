from flask import render_template, request, current_app
from sentence_transformers import SentenceTransformer
import chromadb
import clip
import torch
text_model = SentenceTransformer('all-MiniLM-L6-v2')

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

client = chromadb.PersistentClient(path="chroma_db")
text_collection = client.get_or_create_collection(name="batch_text_embeddings")
image_collection = client.get_or_create_collection(name="batch_image_embeddings")

@current_app.route('/', methods=['GET', 'POST'])
def main():
    results_text = []
    results_image = []

    if request.method == 'POST':
        user_query = request.form['query']

        query_embedding_text = text_model.encode([user_query])

        text_tokens = clip.tokenize([user_query]).to(device)
        with torch.no_grad():
            query_embedding_image = clip_model.encode_text(text_tokens)
        query_embedding_image = query_embedding_image.cpu().numpy().flatten()

        text_results = text_collection.query(
            query_embeddings=query_embedding_text.tolist(),
            n_results=5,
            include=["documents", "metadatas"]
        )
        results_text = list(zip(text_results['documents'][0], text_results['metadatas'][0]))

        image_results = image_collection.query(
            query_embeddings=[query_embedding_image.tolist()],
            n_results=5,
            include=["metadatas"]
        )
        results_image = image_results['metadatas'][0]

    return render_template("base.html", results_text=results_text, results_image=results_image)
