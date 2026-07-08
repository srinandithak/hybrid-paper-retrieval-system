from retrieval.bm25_retrieval import load_papers, build_bm25_index, bm25_search
from retrieval.semantic_search import semantic_search, get_clients

def reciprocal_rank_fusion(semantic_results, bm25_results, k=60):
    scores = {}
    
    for rank, paper in enumerate(semantic_results):
        pid = paper["id"]
        scores[pid] = scores.get(pid, 0) + 1 / (k + rank + 1)
    
    for rank, paper in enumerate(bm25_results):
        pid = paper["id"]
        scores[pid] = scores.get(pid, 0) + 1 / (k + rank + 1)
    
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return sorted_ids


def hybrid_search(query: str, limit: int = 5, pool_size: int = 100):
    supabase, openai_client = get_clients()

    # Need to implement caching for papers and BM25 index to avoid rebuilding on every search
    papers = load_papers(supabase)
    bm25 = build_bm25_index(papers)

    semantic_results = semantic_search(supabase, openai_client, query, limit=pool_size)
    bm25_results = bm25_search(papers, bm25, query, limit=pool_size)

    fused_ids = reciprocal_rank_fusion(semantic_results, bm25_results)
    all_papers = {p["id"]: p for p in papers}
    fused_papers = [all_papers[pid] for pid in fused_ids if pid in all_papers]
    return fused_papers[:limit], len(fused_ids)

if __name__ == "__main__":
    results = hybrid_search("object detection transformers")
    for r in results:
        print(r["title"])
        print("---")