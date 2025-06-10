import json
import os
import requests
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from PIL import Image
import torch
import clip
import ftfy
from io import BytesIO
with open('data/processed/batch_chunks.json', 'r', encoding='utf-8') as f:
    batch_chunks = json.load(f)

with open('data/processed/news_articles.json', 'r', encoding='utf-8') as f:
    news_articles = json.load(f)

print(f"Loaded {len(batch_chunks)} chunks and {len(news_articles)} news articles.")

text_model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [chunk['content'] for chunk in batch_chunks]

text_embeddings = text_model.encode(texts, show_progress_bar=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def get_image_embedding(image_url):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img_preprocessed = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = clip_model.encode_image(img_preprocessed)
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing {image_url}: {e}")
        return None

image_data = []

for i, article in enumerate(tqdm(news_articles)):
    image = article.get('image', {})
    if image and image.get('url'):
        embedding = get_image_embedding(image['url'])
        if embedding is not None:
            image_data.append({
                'embedding': embedding,
                'id': f"img_{i}",
                'metadata': {
                    'image_url': image['url'],
                    'news_title': article['title'],
                    'issue_id': article.get('issue_id', 'unknown'),
                    'issue_url': article.get('issue_url', 'unknown')
                }
            })

print(f"Computed {len(image_data )} image embeddings.")
print("\n=== Saving text embeddings to Chroma DB ===")

client = chromadb.PersistentClient(path="chroma_db")

text_collection = client.get_or_create_collection(name="batch_text_embeddings")

ids = [f"text_chunk_{i}" for i in range(len(text_embeddings))]

metadatas = [{"chunk_id": i} for i in range(len(text_embeddings))]

text_collection.add(
    embeddings=text_embeddings.tolist(),  
    documents=texts,                     
    ids=ids,
    metadatas=metadatas
)

print(f"Inserted {len(text_embeddings)} text embeddings into Chroma")
image_collection = client.get_or_create_collection(name="batch_image_embeddings")

image_collection.add(
    embeddings=[item['embedding'].tolist() for item in image_data],
    ids=[item['id'] for item in image_data],
    metadatas=[item['metadata'] for item in image_data]
)

print(f"Inserted {len(image_data)} image embeddings into Chroma")

print("Text collection count:", text_collection.count())
print("Image collection count:", image_collection.count())
