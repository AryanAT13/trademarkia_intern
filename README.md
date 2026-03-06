# Semantic Search & Fuzzy Caching API

A lightweight, high performance semantic search system built for the 20 Newsgroups dataset. This project implements vector embeddings, fuzzy clustering, and a custom cluster-aware semantic cache from first principles.


### 1. Data Cleaning (The Metadata Trap)
The 20 Newsgroups dataset is riddled with 1990s email headers (`From:`, `Subject:`, `Organization:`). If left intact, embedding models cluster documents based on email domains rather than semantic content. 
* **Decision:** Violently stripped all headers, footers, and quotes using `scikit-learn`'s built-in fetcher. Dropped any remaining documents under 20 characters to ensure only meaningful semantic data was embedded.

### 2. Fuzzy Clustering (GMM vs. K-Means)
The prompt explicitly forbids hard cluster assignments. K-Means is mathematically incapable of fulfilling this requirement.
* **Decision:** Implemented a **Gaussian Mixture Model (GMM)** with 15 components. GMM provides a probability distribution across all clusters for a given document. 
* **Dimensionality Reduction:** To avoid the "curse of dimensionality" which makes GMM covariance matrices unstable on 384-dimensional embeddings, PCA was used to compress vectors to 50 dimensions first.

### 3. The Cluster-Partitioned Cache
A standard dictionary cache degrades to $O(N)$ lookup time as it grows. 
* **Decision:** Built a partitioned cache. Incoming queries are passed through the PCA/GMM pipeline to determine their dominant cluster. The cache *only* searches the memory partition for that specific cluster, drastically reducing the search space.
* **The Tunable Decision (Threshold = 0.85):** Set to 0.85 to balance intent-matching with semantic boundaries. Too high (0.95) mimics an exact-match hash map; too low (0.70) returns false positives across broad topics (e.g., confusing Mac OS with Windows OS).

## Setup & Execution

### Prerequisites
* Python 3.10+
* Mac/Linux Environment

### 1. Install & Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### 2. Prepare the Database & Models
```bash
# Fetches data, cleans it, and populates ChromaDB
python data_prep.py

# Trains the PCA and GMM models for the cache
python clustering.py
```
### 3. Run the FastAPI Service
```bash
python -m uvicorn main:app --reload
Access the API documentation at: http://127.0.0.1:8000/docs
```
