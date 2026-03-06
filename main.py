from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from cache import SemanticCache
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(title="trademarkia internnn")

# We initialize our heavy models and database connections at startup so the API responds instantly to requests.
print("Initializing ChromaDB connection")
DB_PATH = "./chroma_db"
COLLECTION_NAME = "newsgroups_corpus"

try:
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
except Exception as e:
    raise RuntimeError(f"Failed to load ChromaDB: {e}.")

semantic_cache = SemanticCache(threshold=0.85)
print("System Ready.")

class QueryRequest(BaseModel):
    query: str


@app.post("/query")
async def process_query(request: QueryRequest):
    query_text = request.query
    
    cache_result = semantic_cache.check(query_text)
    
    if cache_result.get("cache_hit"):
        return {
            "query": query_text,
            "cache_hit": True,
            "matched_query": cache_result["matched_query"],
            "similarity_score": cache_result["similarity_score"],
            "result": cache_result["result"],
            "dominant_cluster": cache_result["dominant_cluster"]
        }
        
    query_embedding = cache_result["query_embedding"]
    dominant_cluster = cache_result["dominant_cluster"]
    
    # Query ChromaDB for the most relevant document
    db_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=1
    )
    
    if not db_results["documents"] or not db_results["documents"][0]:
        raise HTTPException(status_code=404, detail="No relevant documents found.")
        
    fetched_result = db_results["documents"][0][0]
    
    semantic_cache.add(
        query_text=query_text,
        query_embedding=query_embedding,
        dominant_cluster=dominant_cluster,
        result=fetched_result
    )
    
    return {
        "query": query_text,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": fetched_result,
        "dominant_cluster": dominant_cluster
    }

@app.get("/cache/stats")
async def get_cache_stats():
    return semantic_cache.get_stats()

@app.delete("/cache")
async def flush_cache():
    semantic_cache.clear()
    return {"message": "Cache flushed successfully", "stats": semantic_cache.get_stats()}