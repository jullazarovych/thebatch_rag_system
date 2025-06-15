import os
import re
import time
import json
import random
import logging
import numpy as np
from threading import Lock
from functools import lru_cache
from sentence_transformers import CrossEncoder, SentenceTransformer
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, max_requests=10, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = Lock()

    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            self.requests = [r for r in self.requests if now - r < self.time_window]
            if len(self.requests) >= self.max_requests:
                wait_time = self.time_window - (now - self.requests[0]) + 1
                logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
            self.requests.append(now)


class GeminiProcessor:
    def __init__(self, api_key=None):
        api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.rate_limiter = RateLimiter(max_requests=8, time_window=60)
        self.enabled = True 
        self.relevance_prompt = """
        Analyze the relevance of each piece of content to the user's query on a scale from 0 to 10.

        MUST return ONLY valid JSON in the following format:
        {
        "relevance_score": 8,
        "explanation": "short explanation of relevance",
        "key_matches": ["keywords that matched"]
        }

        DO NOT add any other text, just JSON!
        """

    def make_gemini_request(self, prompt):
        self.rate_limiter.wait_if_needed()
        logger.info(f"Sending prompt to Gemini:\n{prompt}")
        response = self.model.generate_content(prompt)
        response_text = response.text.strip() if hasattr(response, 'text') else None
        logger.info(f"Gemini response:\n{response_text}")
        return response_text

    def safe_json_parse(self, text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            markdown_match = re.search(r"```json\s*\n([\s\S]+?)\n```", text)
            if markdown_match:
                try:
                    return json.loads(markdown_match.group(1))
                except json.JSONDecodeError:
                    pass
        
            json_match = re.search(r"[{\[][\s\S]+?[}\]]", text)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass

            return {
                "relevance_score": 5.0,
                "explanation": "Parse error",
                "key_matches": []
            }
    
    def analyze_relevance(self, query, content, content_type="text"):
        content_text = content if content_type == "text" else str(content.get('news_title', ''))
        prompt = f"{self.relevance_prompt}\n\nQUERY: {query}\nCONTENT: {content_text}"
        response = self.make_gemini_request(prompt)
        return self.safe_json_parse(response)


class QueryExpander:
    def __init__(self):
        self.synonyms = {
            'ai': ['artificial intelligence', 'machine learning', 'neural networks'],
            'ml': ['machine learning', 'ai'],
            'news': ['article', 'report', 'update']
        }

    @lru_cache(maxsize=100)
    def expand_query(self, query):
        words = query.lower().split()
        expanded = set([query])
        for word in words:
            expanded.update(self.synonyms.get(word, []))
        return ' '.join(expanded)


class RerankerModel:
    def __init__(self):
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            logger.warning(f"CrossEncoder init failed: {e}")
            self.cross_encoder = None

    def rerank_results(self, query, results):
        if not self.cross_encoder:
            return results

        pairs = [
            [query, r['content'][:500] if r['type'] == 'text' and r.get('content') else r['meta'].get('news_title', '')]
            for r in results
        ]

        if not pairs:
            logger.warning("No valid pairs found for reranking.")
            return results

        try:
            scores = self.cross_encoder.predict(pairs)
            for i, result in enumerate(results):
                result['rerank_score'] = float(scores[i])
            return sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results


class MultiEmbeddingSearch:
    def __init__(self):
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.technical_model = SentenceTransformer('allenai-specter')

    def classify_query_type(self, query):
        tech_words = ['research', 'study', 'ai', 'ml']
        return 'technical' if any(w in query.lower() for w in tech_words) else 'general'

    def get_embedding(self, text, query_type='general'):
        model = self.technical_model if query_type == 'technical' else self.text_model
        return model.encode(text, normalize_embeddings=True)


class HybridSearcher:
    def calculate_bm25_score(self, query_terms, document, all_documents):
        k1, b = 1.5, 0.75
        doc_len = len(document.split())
        avg_doc_len = np.mean([len(doc.split()) for doc in all_documents if doc]) or 1
        score = 0
        for term in query_terms:
            tf = document.lower().count(term.lower())
            df = sum(1 for doc in all_documents if term.lower() in doc.lower())
            idf = np.log((len(all_documents) - df + 0.5) / (df + 0.5)) if df else 0
            score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len)))
        return score

    def hybrid_search(self, query, results, all_docs):
        terms = query.lower().split()
        for r in results:
            doc = r['content'] if r['type'] == 'text' else r['meta'].get('news_title', '')
            bm25 = self.calculate_bm25_score(terms, doc, all_docs)
            r['hybrid_score'] = 0.6 * r.get('combined_score', 0) + 0.4 * bm25
        return sorted(results, key=lambda x: x['hybrid_score'], reverse=True)


class ResultGrouper:
    def group_results(self, results, batch_chunks=None, news_articles=None):
        grouped = []
        seen_chunks = set()
        for r in results:
            if r['type'] == 'text':
                chunk_id = r['meta'].get('chunk_id')
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    grouped.append({
                        'type': 'grouped',
                        'text_result': r,
                        'image_result': next(
                            (img for img in results if img['type'] == 'image' and img['meta'].get('news_title') == r['meta'].get('news_title')),
                            None
                        ),
                        'news_title': r['meta'].get('news_title'),
                        'news_url': r['meta'].get('issue_url'),
                        'combined_score': r['combined_score']
                    })
            elif r['type'] == 'image':
                already_grouped = any(
                    g.get('image_result') and g['image_result']['meta'].get('image_url') == r['meta'].get('image_url')
                    for g in grouped
                )
                if not already_grouped:
                    grouped.append({
                        'type': 'grouped',
                        'text_result': None,
                        'image_result': r,
                        'news_title': r['meta'].get('news_title'),
                        'news_url': r['meta'].get('issue_url'),
                        'combined_score': r['combined_score']
                    })
        return sorted(grouped, key=lambda x: x['combined_score'], reverse=True)