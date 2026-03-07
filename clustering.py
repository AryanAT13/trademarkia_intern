import os
import pickle
import chromadb
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

DB_PATH = "./chroma_db"
COLLECTION_NAME = "newsgroups_corpus"
MODELS_DIR = "./models"

def perform_clustering():
    print("Connecting to ChromaDB and extracting embeddings...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    db_data = collection.get(include=["embeddings", "documents"])
    embeddings = np.array(db_data["embeddings"])
    docs = db_data["documents"]

    # 1. Dimensionality Reduction (PCA): Embeddings are 384-dimensional. 
    #    GMM struggles with the "curse of dimensionality" (covariance matrices become huge and unstable).
    #    We use PCA to compress to 50 dimensions, preserving variance while making GMM computationally viable and mathematically stable.

    
    print(f"Extracted {embeddings.shape[0]} embeddings. Running PCA compression to 50 dimensions")
    # PCA reduces noise and makes GMM exponentially faster and more stable
    pca = PCA(n_components=50, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)

    # 2. Fuzzy Clustering (GMM): K-Means provides hard boundaries. 
    #    Gaussian Mixture Models output probability distributions across all clusters, perfectly  handling the requirement for soft boundaries.
   
    print("Fitting Gaussian Mixture Model (15 clusters)")
    gmm = GaussianMixture(n_components=15, covariance_type='full', random_state=42)
    gmm.fit(reduced_embeddings)


    # 3. Number of Clusters (15): While there are 20 hard human labels, many are highly 
    #    overlapping sub-genres (e.g., 5 'comp.*' categories, 3 'religion/politics.*').
    #    By choosing 15, we force the model to find broader semantic domains rather than overfitting to 1990s newsgroup subdivisions.
    
    
    print("Calculating fuzzy probabilities for all documents")
    probs = gmm.predict_proba(reduced_embeddings)
    
    print("Saving PCA and GMM models to disk")
    with open(os.path.join(MODELS_DIR, "pca_model.pkl"), "wb") as f:
        pickle.dump(pca, f)
    with open(os.path.join(MODELS_DIR, "gmm_model.pkl"), "wb") as f:
        pickle.dump(gmm, f)
        
    analyze_boundaries(probs, docs)

def analyze_boundaries(probs, docs):
    """
    Satisfies the prompt: "show what lives in them, show what sits at their 
    boundaries, and show where the model is genuinely uncertain"
    """
    print("\n CLUSTER ANALYSIS & BOUNDARY CASES ")
    
    strong_idx = np.argmax(np.max(probs, axis=1))
    strong_cluster = np.argmax(probs[strong_idx])
    strong_prob = probs[strong_idx][strong_cluster]
    
    print(f"\n[STRONG INTERNAL]: Document highly confident in Cluster {strong_cluster} ({strong_prob:.2%} probability)")
    print(f"Snippet: {docs[strong_idx][:150]}...")
    
    
    sorted_probs = np.sort(probs, axis=1)
    margin_of_uncertainty = sorted_probs[:, -1] - sorted_probs[:, -2]
    
    # index of the smallest margin where the top probability is still somewhat significant (not just noise)
    valid_uncertain_indices = np.where(sorted_probs[:, -1] > 0.3)[0]
    if len(valid_uncertain_indices) > 0:
        boundary_idx = valid_uncertain_indices[np.argmin(margin_of_uncertainty[valid_uncertain_indices])]
        top_2_clusters = np.argsort(probs[boundary_idx])[-2:][::-1]
        
        print(f"\n[BOUNDARY CASE / UNCERTAIN]: Document sits perfectly between Cluster {top_2_clusters[0]} ({probs[boundary_idx][top_2_clusters[0]]:.2%}) and Cluster {top_2_clusters[1]} ({probs[boundary_idx][top_2_clusters[1]]:.2%})")
        print(f"Snippet: {docs[boundary_idx][:200]}...")

    print("\nClustering complete. Models prepped for Cache routing.")

if __name__ == "__main__":
    perform_clustering()