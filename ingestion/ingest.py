import arxiv
import os
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI
import json

load_dotenv()

# Paper storage in Supabase
def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)

def store_paper(supabase, paper: dict):
    response = supabase.table("papers").upsert({
        "id": paper["id"],
        "title": paper["title"],
        "abstract": paper["abstract"],
        "authors": paper["authors"],
        "published": paper["published"],
        "url": paper["url"]
    }).execute()
    return response

#Papers from arxiv
def fetch_papers(query: str, max_results: int = 50):
    client = arxiv.Client(delay_seconds=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers = []
    for result in client.results(search):
        papers.append({
            "id": result.entry_id.split("/")[-1],
            "title": result.title,
            "abstract": result.summary,
            "authors": [a.name for a in result.authors],
            "published": str(result.published.date()),
            "url": result.entry_id
        })
    
    return papers

#Embedding to store in Supabase
def generate_embedding(openai_client, text: str) -> list:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def store_embedding(supabase, paper_id: str, embedding: list):
    supabase.table("papers").update({
        "embedding": embedding
    }).eq("id", paper_id).execute()

# extracting important info from abstract
def extract_structure(openai_client, abstract: str) -> dict:
    prompt = f"""Extract structured information from this research paper abstract.
Return ONLY valid JSON with these exact keys, no other text:

{{
    "methods": ["list of methods or architectures used"],
    "datasets": ["list of datasets mentioned"],
    "metrics": {{"metric_name": "reported_value"}},
    "keywords": ["list of 3-5 key topics"]
}}

If a field has no information, return an empty list or object for it.

Abstract:
{abstract}"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    raw = response.choices[0].message.content
    return json.loads(raw)

def store_extraction(supabase, paper_id: str, extraction: dict):
    supabase.table("papers").update({
        "methods": extraction.get("methods", []),
        "datasets": extraction.get("datasets", []),
        "metrics": extraction.get("metrics", {}),
        "keywords": extraction.get("keywords", [])
    }).eq("id", paper_id).execute()

def paper_exists(supabase, paper_id: str) -> bool:
    result = supabase.table("papers").select("id").eq("id", paper_id).execute()
    return len(result.data) > 0


if __name__ == "__main__":
    supabase = get_supabase_client()
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    papers = fetch_papers("computer vision object detection", max_results=50)
        
    for p in papers:
        if paper_exists(supabase, p["id"]):
            print(f"Skipping (exists): {p['title']}")
            continue
        
        store_paper(supabase, p)
        embedding = generate_embedding(openai_client, p["abstract"])
        store_embedding(supabase, p["id"], embedding)
        extraction = extract_structure(openai_client, p["abstract"])
        store_extraction(supabase, p["id"], extraction)
        print(f"Done: {p['title']}")