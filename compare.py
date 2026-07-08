import openai
import re 

# Builds prompt to ask LLM to answer query only using papers in database 
def build_compare_prompt(query: str, papers: list[dict]) -> str:
    paper_blocks = []
    for p in papers:
        norm_id = normalize_id(p["id"])
        paper_blocks.append(
            f"[{norm_id}] Title: {p['title']}\n"
            f"Abstract: {p['abstract']}\n"
        )
    papers_text = "\n\n".join(paper_blocks)

    prompt = f"""You are answering a question using a set of research papers. If the question naturally invites comparing multiple papers, compare them explicitly and note points of agreement or disagreement. If it's about a single fact or paper, answer directly without forcing a comparison.

Question: {query}

Here are the only papers you may use. Each has an ID in brackets:

{papers_text}

Instructions:
- Answer using ONLY the papers listed above.
- For every claim you make, cite the paper ID it came from, like this: [paper_id].
- Do not cite any ID that is not in the list above.
- If the provided papers don't fully answer the question, say so explicitly rather than filling gaps with outside knowledge.
"""
    return prompt

# If result fails to follow rules this is second try 
def build_retry_prompt(query: str, papers: list[dict], invalid_citations: list[str]) -> str:
    base_prompt = build_compare_prompt(query, papers)
    valid_ids = [normalize_id(p["id"]) for p in papers]

    retry_note = f"""

IMPORTANT: Your previous attempt cited paper ID(s) that do not exist in the provided list: {invalid_citations}.
You may ONLY cite these exact IDs: {valid_ids}.
Do not invent or reference any ID outside this list. Re-generate your full answer following this constraint strictly.
"""
    return base_prompt + retry_note

# Compares paper
def generate_comparison(prompt: str) -> str:
    client = openai.OpenAI()  
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

#Ensures that the citations in the LLM's response are valid and grounded in the retrieved papers
def generate_grounded_comparison(query: str, papers: list[dict], max_retries: int = 1) -> dict:
    retrieved_ids = {p["id"] for p in papers}

    prompt = build_compare_prompt(query, papers)
    attempt = 0
    llm_response = None
    validation = None

    while attempt <= max_retries:
        llm_response = generate_comparison(prompt)
        cited_ids = extract_citations(llm_response)
        validation = validate_citations(cited_ids, retrieved_ids)

        if validation["is_fully_grounded"]:
            break

        # build a stricter prompt for the next attempt
        prompt = build_retry_prompt(query, papers, validation["invalid_citations"])
        attempt += 1

    return {
        "comparison": llm_response,
        "validation": validation,
        "attempts": attempt + 1
    }


def extract_citations(llm_text: str) -> list[str]:
    return re.findall(r"\[([^\[\]]+)\]", llm_text)

def validate_citations(cited_ids: list[str], retrieved_ids: set[str]) -> dict:
    normalized_retrieved = {normalize_id(rid) for rid in retrieved_ids}
    valid = [cid for cid in cited_ids if normalize_id(cid) in normalized_retrieved]
    invalid = [cid for cid in cited_ids if normalize_id(cid) not in normalized_retrieved]
    return {
        "valid_citations": valid,
        "invalid_citations": invalid,
        "is_fully_grounded": len(invalid) == 0
    }


def normalize_id(paper_id: str) -> str:
    return re.sub(r"v\d+$", "", paper_id)