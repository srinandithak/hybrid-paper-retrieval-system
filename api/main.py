from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from retrieval.hybrid_retrieval import hybrid_search
from compare import generate_grounded_comparison

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",           # local Vite dev server
        "https://hybrid-paper-retrieval-system.onrender.com/",     
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/compare")
def compare(query: str, limit: int = 8):
    papers, _ = hybrid_search(query, limit=limit)
    result = generate_grounded_comparison(query, papers)

    return {
        "query": query,
        "comparison": result["comparison"],
        "papers_used": [p["id"] for p in papers],
        "validation": result["validation"],
        "attempts": result["attempts"]
    }