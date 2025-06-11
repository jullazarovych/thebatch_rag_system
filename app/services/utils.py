import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm != 0 else embedding

def cosine_similarity_score(a, b):
    if a is None or b is None:
        return 0.0
    return cosine_similarity([a], [b])[0][0]

def safe_get(d, *keys, default=None):
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d
