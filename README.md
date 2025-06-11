# Multimodal RAG System — News Retrieval Demo
This repository implements a Retrieval-Augmented Generation (RAG) system tailored for multimodal search across articles and images from the "The Batch" newsletter by DeepLearning.ai. It integrates embedding models, reranking, Gemini-based reasoning, and a clean web UI using Flask.

### Overview
This project demonstrates a retrieval-augmented generation system for searching and analyzing multimodal content — i.e., both text and associated images in news articles. It uses:
- Sentence and CLIP embeddings for retrieval;
- Gemini for semantic relevance and answer synthesis;
- ChromaDB for vector storage;
- Flask for the web interface.

### Architecture
```
    User[User Query]
    User -->|Text & Image Embeddings| Embed
    Embed -->|Query Vector| Search[ChromaDB (text & image collections)]
    Search -->|Top-K Matches| RAG[RAG Pipeline]
    RAG -->|CLIP + Cosine + Gemini| Scorer
    Scorer -->|Reranked Results| UI[Flask Web UI]
    UI -->|Gemini-generated Answer| User
```

### Data Collection (data_collection.py)
This module implements a fully automated scraper and preprocessor for DeepLearning.ai’s The Batch newsletter.
Class: BatchDataCollector
Scrapes all recent newsletter issues and extracts:
- News article titles
- Images and captions
- Text content (including paragraphs and lists)
- Splits article content into overlapping chunks using RecursiveCharacterTextSplitter.
- Stores data in:
  - data/raw/batch_articles.json: all raw issue data
  - data/processed/batch_chunks.json: processed text chunks, each with metadata (e.g., chunk_id, issue URL, title)
  - data/processed/news_articles.json: structured per-article data with images.

### Approach
- Using a custom synonym-based QueryExpander to semantically broaden the input query.
- Multimodal Embedding Generation
  - text queries: encoded via SentenceTransformer (MiniLM + Specter for technical context).
  - image queries: encoded via OpenAI's CLIP.
- Retrieval with ChromaDB — retrieved top-N results from both text_collection and image_collection.
- Scoring
  <For each result:
  < - Cosine similarity between query and document/image.
  < - Gemini Relevance Scoring (0–10) via LLM prompt.
  < Combined Score: weighted average of text, image, and Gemini relevance.
  < Hybrid Search + BM25-inspired Weighting — BM25 scoring approximated using TF-IDF heuristic for better lexical relevance.
  < Each result is scored based on:
  ```
  combined_score = 0.3 * score_text + 0.3 * score_image + 0.4 * gemini_relevance
  ```
- Reranking via CrossEncoder (MiniLM) is applied to the top 12 text-based or title-bearing results using semantic similarity between the query and each item's content or title.
- A summary answer is generated using Google Gemini by prompting it with content from the top 3 ranked results. The model returns an answer, confidence level, and recommended follow-up queries.
- Rate Limiting: Gemini API is called with a limit of 8 requests/minute (implemented via RateLimiter) (If the key is missing, Gemini analysis simply boils down, and the system continues to work without LLM).
- UI — top 5 grouped results are displayed in a Bootstrap-powered UI with titles, relevance, and optional images.

### Tech Stack
Language	Python 3.12
Backend	Flask
Frontend	HTML, Bootstrap
Vector DB	ChromaDB
Embeddings	SentenceTransformers, CLIP
Reranking	CrossEncoder (MiniLM)
LLM	Google Gemini 1.5
Deployment	Local / Docker-ready

### Project Structure
```
thebatch_rag_system/
├── app/
│   ├── __init__.py
│   ├── view.py
│   └── services/
│       ├── __init__.py
│       ├── search.py                    # Core logic for handling user queries, embedding, retrieval, reranking, and rendering.
│       ├── models.py                    # Model & data initialization: loads SentenceTransformers, CLIP, ChromaDB collections, etc.
│       ├── processors.py                 Key search components.
│       ├── utils.py                     # Helper utilities (L2-normalization, cosine similarity between embeddings, dict traversal for nested fields)
├── static/                   # Empty for now
│   ├── css
│   ├── images
│   └── js
├── scripts/
│   ├── datacollection.py     # Extract newі fron The Batch
│   ├── embedding.py          # Preprocessing script
│   └── test_emb.py           # Visualisation of embeddings
├── templates/
│   └── base.html             # Main UI template
├── data/
│   └── processed/            # Contains batch_chunks.json, news_articles.json
├── chroma_db/                # ChromaDB storage
├── .flaskenv                 # Flask
├── .env                      # Gemini API key
├── requirements.txt
├── run.py
└── README.md
```

### Installation
Prerequisites:
- Python 3.12 
- Git
- pip

1. Clone the repo:
```
git clone https://github.com/yourusername/thebatch_rag_system.git
cd thebatch_rag_system
```
2. Create virtual environment:
```
python -m venv .venv
source .venv/bin/activate      # On Linux/macOS
.venv\Scripts\activate         # On Windows
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Environment Variables
Create a .env file in the root directory with:
```
GEMINI_API_KEY=your_gemini_api_key
```
5. Run the Application
```
python scripts/datacallection.py #extract news and break them into chunks
```
Scrapes latest 20 issues
- Extracts and chunks content
- Saves to:
  - data/raw/batch_articles.json
  - data/processed/batch_chunks.json
  - data/processed/news_articles.json
```
python scripts/embedding.py      # Prepare embeddings and insert into ChromaDB
python run.py
```
#### Gemini is used to:
Assess relevance of retrieved items
Generate synthesized answers from top results
Gemini calls are rate-limited to 8 requests per minute to comply with the API quota. If limits are hit, requests will wait ~60 seconds.
If you don’t provide a GEMINI_API_KEY, the system will gracefully skip Gemini processing.          
