from fastapi import FastAPI
from supabase import create_client
from retrieval.hybrid_retrieval import hybrid_search

app = FastAPI()

@app.get("/search")
def search(query: str, limit: int = 10, offset: int = 0):
    results, total_candidates = hybrid_search(query, limit=limit + offset, pool_size=100)
    paginated = results[offset:offset + limit]
    return {
        "results": paginated,
        "limit": limit,
        "offset": offset,
        "total": total_candidates
    }