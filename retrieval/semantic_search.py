import os
from openai import OpenAI
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

def get_clients():
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return supabase, openai_client

def semantic_search(supabase, openai_client, query: str, limit: int = 5):
    query_embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    result = supabase.rpc("match_papers", {
        "query_embedding": query_embedding,
        "match_count": limit
    }).execute()

    return result.data

if __name__ == "__main__":
    supabase, openai_client = get_clients()
    results = semantic_search(supabase, openai_client, "object detection with transformers")
    for r in results:
        print(r["title"])
        print(r["similarity"])
        print("---")