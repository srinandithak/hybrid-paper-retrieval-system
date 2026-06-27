from supabase import create_client
from retrieval.hybrid_retrieval import hybrid_search


if __name__ == "__main__":
    query = input("Enter your search")
    hybrid_result = hybrid_search(query, limit=5)