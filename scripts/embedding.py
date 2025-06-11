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
import re

with open('data/processed/batch_chunks.json', 'r', encoding='utf-8') as f:
    batch_chunks = json.load(f)

with open('data/processed/news_articles.json', 'r', encoding='utf-8') as f:
    news_articles = json.load(f)

print(f"Loaded {len(batch_chunks)} chunks and {len(news_articles)} news articles.")

def create_chunk_to_news_mapping(chunks, articles):
    chunk_to_news = {}
    
    for chunk_idx, chunk in enumerate(chunks):
        chunk_content = chunk['content'].lower()
        
        best_match = None
        best_score = 0
        
        for article_idx, article in enumerate(articles):
            title_words = set(re.findall(r'\w+', article['title'].lower()))
            chunk_words = set(re.findall(r'\w+', chunk_content))
            
            title_overlap = len(title_words.intersection(chunk_words))
            
            content_overlap = 0
            if 'content' in article and article['content']:
                content_words = set(re.findall(r'\w+', article['content'].lower()))
                content_overlap = len(content_words.intersection(chunk_words))
            
            total_score = title_overlap * 2 + content_overlap  
            
            if total_score > best_score:
                best_score = total_score
                best_match = article_idx
        
        if best_match is not None and best_score > 2: 
            chunk_to_news[chunk_idx] = {
                'article_index': best_match,
                'match_score': best_score,
                'news_title': articles[best_match]['title'],
                'issue_id': articles[best_match].get('issue_id', 'unknown'),
                'issue_url': articles[best_match].get('issue_url', 'unknown')
            }
    
    return chunk_to_news

print("Creating chunk-to-news mapping...")
chunk_to_news_map = create_chunk_to_news_mapping(batch_chunks, news_articles)
print(f"Mapped {len(chunk_to_news_map)} chunks to news articles.")

text_model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [chunk['content'] for chunk in batch_chunks]

print("Computing text embeddings...")
text_embeddings = text_model.encode(texts, show_progress_bar=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def get_image_embedding(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img_preprocessed = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = clip_model.encode_image(img_preprocessed)
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing {image_url}: {e}")
        return None

print("Computing image embeddings...")
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
                    'issue_url': article.get('issue_url', 'unknown'),
                    'article_index': i,
                    'type': 'image'
                }
            })

print(f"Computed {len(image_data)} image embeddings.")

print("\n=== Saving embeddings to Chroma DB ===")
client = chromadb.PersistentClient(path="chroma_db")

try:
    client.delete_collection(name="batch_text_embeddings")
    client.delete_collection(name="batch_image_embeddings")
    print("Deleted existing collections.")
except:
    print("No existing collections to delete.")

text_collection = client.get_or_create_collection(name="batch_text_embeddings")
image_collection = client.get_or_create_collection(name="batch_image_embeddings")

print("Preparing text embeddings data...")
text_ids = []
text_metadatas = []

for i, (chunk, embedding) in enumerate(zip(batch_chunks, text_embeddings)):
    text_ids.append(f"text_chunk_{i}")
    
    metadata = {
        "chunk_id": i,
        "chunk_length": len(chunk['content']),
        "type": "text"
    }
    
    if i in chunk_to_news_map:
        news_info = chunk_to_news_map[i]
        metadata.update({
            "news_title": news_info['news_title'],
            "issue_id": news_info['issue_id'],
            "issue_url": news_info['issue_url'],
            "article_index": news_info['article_index'],
            "match_score": news_info['match_score']
        })
    
    text_metadatas.append(metadata)

print("Saving text embeddings...")
text_collection.add(
    embeddings=text_embeddings.tolist(),
    documents=texts,
    ids=text_ids,
    metadatas=text_metadatas
)

print(f"Inserted {len(text_embeddings)} text embeddings into Chroma")

print("Saving image embeddings...")
if image_data:
    image_collection.add(
        embeddings=[item['embedding'].tolist() for item in image_data],
        ids=[item['id'] for item in image_data],
        metadatas=[item['metadata'] for item in image_data]
    )
    print(f"Inserted {len(image_data)} image embeddings into Chroma")
else:
    print("No image embeddings to save.")

print("\n=== Verification ===")
print("Text collection count:", text_collection.count())
print("Image collection count:", image_collection.count())

