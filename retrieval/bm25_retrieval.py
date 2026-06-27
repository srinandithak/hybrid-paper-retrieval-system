from rank_bm25 import BM25Okapi
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

def load_papers(supabase):
    result = supabase.table("papers").select("id, title, abstract").execute()
    return result.data

def build_bm25_index(papers):
    corpus = [p["abstract"].lower().split() for p in papers]
    return BM25Okapi(corpus)

def bm25_search(papers, bm25, query: str, limit: int = 5):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
    
    return [{"id": papers[i]["id"], "title": papers[i]["title"], "score": scores[i]} for i in top_indices]

if __name__ == "__main__":
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    papers = load_papers(supabase)
    bm25 = build_bm25_index(papers)
    
    results = bm25_search(papers, bm25, "object detection transformers", limit=5)
    for r in results:
        print(r["title"])
        print(r["score"])
        print("---")