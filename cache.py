import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

# 1. Cluster-Based Optimization: Instead of a flat list where lookup time grows O(N), our cache is a dictionary partitioned by the 15 GMM clusters.
#    We predict the incoming query's dominant cluster and ONLY search that partition. 
#    This massively reduces the search space and ensures O(N/K) lookup time.


class SemanticCache:
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.store = {i: [] for i in range(15)} 
        self.stats = {
            "total_entries": 0,
            "hit_count": 0,
            "miss_count": 0
        }

        # 2. The Tunable Decision (SIMILARITY_THRESHOLD = 0.85): 
        #    This is the heart of the cache. 
        #    - Too high (e.g., 0.95): The cache acts like an exact-match hash map, defeating the purpose of semantic caching. Hit rates will plummet.
        #    - Too low (e.g., 0.70): The cache returns false positives (e.g., returning a response about "Mac OS" for a query about "Windows OS" because both are OS queries).
        #    - 0.85 balances intent-matching (ignoring typos/phrasing) while respecting hard semantic boundaries. It reveals that the system prioritizes accuracy  over raw hit-rate speed.

        
        print("Loading Embedding, PCA, and GMM models into Cache...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        with open("./models/pca_model.pkl", "rb") as f:
            self.pca = pickle.load(f)
        with open("./models/gmm_model.pkl", "rb") as f:
            self.gmm = pickle.load(f)

    def _cosine_similarity(self, vec1, vec2):
        """Calculates cosine similarity between two 1D vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _get_dominant_cluster(self, query_embedding):
        """Passes the query through PCA and GMM to find its dominant cluster."""
        # Reshape for sklearn (1 sample, n features)
        embedding_reshaped = np.array(query_embedding).reshape(1, -1)
        reduced = self.pca.transform(embedding_reshaped)
        probs = self.gmm.predict_proba(reduced)
        dominant_cluster = int(np.argmax(probs[0]))
        return dominant_cluster

    def check(self, query_text):
        """Checks the cache for a semantic match."""
        query_embedding = self.embedder.encode(query_text)
        cluster_id = self._get_dominant_cluster(query_embedding)
        
        # ONLY search within the dominant cluster's partition
        partition = self.store[cluster_id]
        
        best_match = None
        highest_score = 0.0
        
        for entry in partition:
            score = self._cosine_similarity(query_embedding, entry["embedding"])
            if score > highest_score:
                highest_score = score
                best_match = entry
                
        if highest_score >= self.threshold:
            self.stats["hit_count"] += 1
            return {
                "cache_hit": True,
                "matched_query": best_match["query_text"],
                "similarity_score": round(float(highest_score), 4),
                "result": best_match["result"],
                "dominant_cluster": cluster_id
            }
            
        self.stats["miss_count"] += 1
        return {
            "cache_hit": False,
            "query_embedding": query_embedding,
            "dominant_cluster": cluster_id
        }

    def add(self, query_text, query_embedding, dominant_cluster, result):
        """Adds a new entry to the appropriate cluster partition."""
        entry = {
            "query_text": query_text,
            "embedding": query_embedding,
            "result": result
        }
        self.store[dominant_cluster].append(entry)
        self.stats["total_entries"] += 1

    def get_stats(self):
        """Calculates and returns current cache performance metrics."""
        total_requests = self.stats["hit_count"] + self.stats["miss_count"]
        hit_rate = self.stats["hit_count"] / total_requests if total_requests > 0 else 0.0
        
        return {
            "total_entries": self.stats["total_entries"],
            "hit_count": self.stats["hit_count"],
            "miss_count": self.stats["miss_count"],
            "hit_rate": round(hit_rate, 3)
        }

    def clear(self):
        """Flushes the cache and resets stats."""
        self.store = {i: [] for i in range(15)}
        self.stats = {
            "total_entries": 0,
            "hit_count": 0,
            "miss_count": 0
        }