from flask import render_template, request, current_app
from sentence_transformers import SentenceTransformer
import chromadb
import clip
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def cosine(a, b):
    return cosine_similarity([a], [b])[0][0]

text_model = SentenceTransformer('all-MiniLM-L6-v2')

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

client = chromadb.PersistentClient(path="chroma_db")
text_collection = client.get_or_create_collection(name="batch_text_embeddings")
image_collection = client.get_or_create_collection(name="batch_image_embeddings")
@current_app.route('/', methods=['GET', 'POST'])
def main():
    combined_results = []

    if request.method == 'POST':
        user_query = request.form['query']

        query_embedding_text = text_model.encode(user_query, normalize_embeddings=True)  
        query_embedding_text = query_embedding_text.tolist() 

        text_tokens = clip.tokenize([user_query]).to(device)
        with torch.no_grad():
            query_embedding_image = clip_model.encode_text(text_tokens)
        query_embedding_image = query_embedding_image.cpu().numpy().flatten()
        query_embedding_image = normalize_embedding(query_embedding_image).tolist()

        text_results = text_collection.query(
            query_embeddings=[query_embedding_text],
            n_results=10,
            include=["documents", "metadatas", "embeddings"]
        )

        image_results = image_collection.query(
            query_embeddings=[query_embedding_image],
            n_results=10,
            include=["metadatas", "embeddings"]
        )

        for doc, meta, emb in zip(
            text_results['documents'][0],
            text_results['metadatas'][0],
            text_results['embeddings'][0]
        ):
            score_text = cosine(query_embedding_text, emb)
            combined_results.append({
                'type': 'text',
                'content': doc,
                'meta': meta,
                'score_text': score_text,
                'score_image': 0.0
            })

        for meta, emb in zip(
            image_results['metadatas'][0],
            image_results['embeddings'][0]
        ):
            score_image = cosine(query_embedding_image, emb)
            combined_results.append({
                'type': 'image',
                'content': '',
                'meta': meta,
                'score_text': 0.0,
                'score_image': score_image
            })

        alpha = 0.5
        beta = 0.5

        for item in combined_results:
            item['combined_score'] = alpha * item['score_text'] + beta * item['score_image']

        combined_results = sorted(combined_results, key=lambda x: x['combined_score'], reverse=True)
        combined_results = combined_results[:5]

    return render_template("base.html", combined_results=combined_results)
