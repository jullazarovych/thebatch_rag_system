import chromadb
from chromadb.config import Settings
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

n_records = 200

client = chromadb.PersistentClient(path="chroma_db")
text_collection = client.get_collection("batch_text_embeddings")

results = text_collection.get(include=["embeddings", "metadatas"], limit=n_records)

embeddings = np.array(results["embeddings"])

tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1])

for i, metadata in enumerate(results["metadatas"]):
    plt.annotate(metadata.get("news_title", ""), (X_embedded[i, 0], X_embedded[i, 1]), fontsize=8)

plt.title("t-SNE visualization of text embeddings")
plt.show()
