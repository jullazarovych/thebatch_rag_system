import os
import json
import torch
import clip
import chromadb
import logging
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.services.utils import normalize_embedding
from app.services.processors import (
    QueryExpander, RerankerModel, MultiEmbeddingSearch,
    HybridSearcher, ResultGrouper, GeminiProcessor
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

query_expander = QueryExpander()
reranker = RerankerModel()
multi_embedding = MultiEmbeddingSearch()
hybrid_searcher = HybridSearcher()
result_grouper = ResultGrouper()
try:
    gemini_processor = GeminiProcessor()
except Exception as e:
    logger.warning(f"Gemini disabled: {e}")
    gemini_processor = None

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    logger.info(f"CLIP model loaded on {device}")
except Exception as e:
    logger.error(f"Failed to load CLIP: {e}")
    clip_model = None
    preprocess = None

try:
    client = chromadb.PersistentClient(path="chroma_db")
    text_collection = client.get_or_create_collection(name="batch_text_embeddings")
    image_collection = client.get_or_create_collection(name="batch_image_embeddings")
    logger.info("ChromaDB collections loaded")
except Exception as e:
    logger.error(f"Failed to connect to ChromaDB: {e}")
    text_collection = None
    image_collection = None

try:
    with open('data/processed/batch_chunks.json', 'r', encoding='utf-8') as f:
        batch_chunks = json.load(f)
    with open('data/processed/news_articles.json', 'r', encoding='utf-8') as f:
        news_articles = json.load(f)
    logger.info("Data files loaded")
except FileNotFoundError as e:
    logger.warning(f"Data file missing: {e}")
    batch_chunks = []
    news_articles = []
