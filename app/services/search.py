from flask import render_template
from app.services.models import (
    query_expander, multi_embedding, clip_model, device,
    text_collection, image_collection, result_grouper,
    gemini_processor, hybrid_searcher, reranker,
    batch_chunks, news_articles
)
from app.services.utils import normalize_embedding, cosine_similarity_score
import torch
import numpy as np
import logging
import clip


logger = logging.getLogger(__name__)

def handle_query(user_query):
    expanded_query = query_expander.expand_query(user_query)
    query_type = multi_embedding.classify_query_type(user_query)
    query_embedding_text = multi_embedding.get_embedding(expanded_query, query_type)

    if clip_model:
        text_tokens = clip.tokenize([expanded_query]).to(device)
        with torch.no_grad():
            query_embedding_image = clip_model.encode_text(text_tokens)
        query_embedding_image = normalize_embedding(query_embedding_image.cpu().numpy().flatten()).tolist()
    else:
        query_embedding_image = np.random.rand(512).tolist()

    combined_results = []
    all_documents = []

    if text_collection is not None:
        try:
            text_results = text_collection.query(
                query_embeddings=[query_embedding_text],
                n_results=15,
                include=["documents", "metadatas", "embeddings"]
            )

            for i, (doc, meta, emb) in enumerate(zip(
                text_results['documents'][0],
                text_results['metadatas'][0],
                text_results['embeddings'][0]
            )):
                score_text = cosine_similarity_score(query_embedding_text, emb)

                if i < 5:
                    gemini_analysis = gemini_processor.analyze_relevance(user_query, doc, "text")
                else:
                    gemini_analysis = {"relevance_score": 5.0, "explanation": "Not analyzed", "key_matches": []}

                combined_results.append({
                    'type': 'text',
                    'content': doc,
                    'meta': meta,
                    'score_text': score_text,
                    'score_image': 0.0,
                    'gemini_relevance': gemini_analysis['relevance_score'] / 10.0,
                    'gemini_explanation': gemini_analysis.get('explanation', 'Not available'),
                    'gemini_matches': gemini_analysis.get('key_matches', []),
                })
                all_documents.append(doc)
        except Exception as e:
            logger.error(f"Text search error: {e}")

    if image_collection is not None:
        try:
            image_results = image_collection.query(
                query_embeddings=[query_embedding_image],
                n_results=10,
                include=["metadatas", "embeddings"]
            )

            for i, (meta, emb) in enumerate(zip(
                image_results['metadatas'][0],
                image_results['embeddings'][0]
            )):
                score_image = cosine_similarity_score(query_embedding_image, emb)

                if i < 3:
                    gemini_analysis = gemini_processor.analyze_relevance(user_query, meta, "image")
                else:
                    gemini_analysis = {"relevance_score": 5.0, "explanation": "Not analyzed", "key_matches": []}

                combined_results.append({
                    'type': 'image',
                    'content': '',
                    'meta': meta,
                    'score_text': 0.0,
                    'score_image': score_image,
                    'gemini_relevance': gemini_analysis['relevance_score'] / 10.0,
                    'gemini_explanation': gemini_analysis['explanation'],
                    'gemini_matches': gemini_analysis['key_matches']
                })
        except Exception as e:
            logger.error(f"Image search error: {e}")

    alpha, beta, gamma = 0.3, 0.3, 0.4
    for item in combined_results:
        item['combined_score'] = (
            alpha * item['score_text'] +
            beta * item['score_image'] +
            gamma * item['gemini_relevance']
        )
    combined_results = hybrid_searcher.hybrid_search(user_query, combined_results, all_documents)
    combined_results = reranker.rerank_results(user_query, combined_results)

    combined_results = combined_results[:5]

    grouped_results = result_grouper.group_results(
        combined_results, batch_chunks, news_articles
    )

    generated_answer = gemini_processor.generate_answer(user_query, combined_results[:3])

    search_stats = {
        'original_query': user_query,
        'expanded_query': expanded_query,
        'query_type': query_type,
        'total_results': len(combined_results),
        'grouped_results': len(grouped_results),
        'text_results': len([r for r in combined_results if r['type'] == 'text']),
        'image_results': len([r for r in combined_results if r['type'] == 'image']),
        'avg_gemini_relevance': np.mean([r['gemini_relevance'] for r in combined_results]) if combined_results else 0,
        'gemini_enabled': getattr(gemini_processor, 'enabled', False)
    }


    logger.info(f"Search completed successfully for query: {user_query}")

    return render_template(
        "base.html",
        combined_results=combined_results,
        grouped_results=grouped_results,
        generated_answer=generated_answer,
        search_stats=search_stats,
        user_query=user_query  
    )
