import os
from sklearn.datasets import fetch_20newsgroups
import chromadb
import chromadb.errors
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DB_PATH = "./chroma_db"
COLLECTION_NAME = "newsgroups_corpus"
BATCH_SIZE = 256 

# 1. Data Cleaning: We use scikit-learn's built-in remove parameter to strip headers', 'footers', and 'quotes'.
#    If we retain headers, embeddings will overfit to metadata (e.g., email routing, domains) rather than semantic content.
#    Quotes are removed to prevent heavy cross-contamination of text from replies.

def prepare_and_embed_data():
    print("Fetching and cleaning dataset...")
    dataset = fetch_20newsgroups(
        subset='all', 
        remove=('headers', 'footers', 'quotes')
    )
    
    docs = dataset.data
    labels = dataset.target 
    
    cleaned_docs = []
    cleaned_labels = []
    doc_ids = []
    
    print("Filtering out empty or meaningless documents...")
    for i, doc in enumerate(docs):
        clean_text = doc.strip()
        if len(clean_text) > 20: 
            cleaned_docs.append(clean_text)
            cleaned_labels.append(int(labels[i]))
            doc_ids.append(str(i))
            
    print(f"Retained {len(cleaned_docs)} viable documents out of {len(docs)}.")

    # 2. Embedding Model: 'all-MiniLM-L6-v2'. It is a highly optimized, lightweight sentence transformer. 
    #    Given the ~20k document corpus, we need a model that balances semantic accuracy with fast local inference (no GPU required).

    print("Initializing embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)

    # 3. Vector Database: ChromaDB. It is a lightweight, serverless, and persistent local vector store. 
    #    It perfectly fits the "lightweight system" requirement without needing external infrastructure like Postgres/pgvector.
    
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        pass # Collection didn't exist yet
    except ValueError:
        pass # Fallback for older versions
        
    collection = client.create_collection(name=COLLECTION_NAME)

    print("Generating embeddings and upserting to vector database...")
    for i in tqdm(range(0, len(cleaned_docs), BATCH_SIZE)):
        batch_docs = cleaned_docs[i:i + BATCH_SIZE]
        batch_ids = doc_ids[i:i + BATCH_SIZE]
        batch_metadata = [{"original_label": label} for label in cleaned_labels[i:i + BATCH_SIZE]]
        
        batch_embeddings = model.encode(batch_docs).tolist()
        
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_docs,
            metadatas=batch_metadata
        )

    print(f"Successfully embedded and stored {collection.count()} documents in ChromaDB.")

if __name__ == "__main__":
    prepare_and_embed_data()