import re
from typing import List, Tuple
from langchain_core.language_models.base import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# === 1. Extract all [label](url) links ===
def extract_references(markdown_text: str) -> List[Tuple[str, str]]:
    pattern = r'\[([^\]]+)\]\((https?://[^\)]+)\)'
    matches = re.findall(pattern, markdown_text)

    seen = set()
    refs = []
    for label, url in matches:
        if url not in seen:
            refs.append((label, url))
            seen.add(url)
    return refs


# === 2. Replace [Label](url) with [n] ===
def replace_inline_references(markdown_text: str, refs: List[Tuple[str, str]]) -> str:
    new_md = markdown_text
    for idx, (_, url) in enumerate(refs, 1):
        new_md = re.sub(rf'\[([^\]]+)\]\({re.escape(url)}\)', f'[{idx}]', new_md)
    return new_md


# === 3. Rephrase citation descriptions using LLM ===
def rephrase_citations(refs: List[Tuple[str, str]], llm: BaseLanguageModel) -> List[str]:
    prompt = PromptTemplate.from_template("""
You are a smart assistant. Rewrite the following list of citation URLs into short, professional source descriptions.

RULES:
- Do NOT change the links.
- Just provide a short description (max 10 words) of what the link refers to.
- If the source seems low quality (e.g., blogspot, medium), write "Unreliable source" instead.

Example:
Input:
https://www.nytimes.com/2023/01/01/ai-google-news.html

Output:
Google AI announcement (NYTimes) – [Link](https://www.nytimes.com/...)

Now rewrite these links:

{urls}
""")

    # Convert refs to raw URLs
    urls_text = "\n".join([url for _, url in refs])
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(urls=urls_text)

    # Split by line and clean
    return [line.strip() for line in result.strip().split("\n") if line.strip()]


# === 4. Full citation processor ===
def run_citation(markdown_text: str, llm: BaseLanguageModel) -> str:
    refs = extract_references(markdown_text)
    if not refs:
        return markdown_text

    numbered_text = replace_inline_references(markdown_text, refs)
    rephrased_refs = rephrase_citations(refs, llm)

    # Clean existing References section if any
    numbered_text = re.sub(r'\n## References[\s\S]+$', '', numbered_text).strip()

    # Append new references
    references_section = "\n\n## References\n"
    for idx, desc in enumerate(rephrased_refs, 1):
        references_section += f"{idx}. {desc}\n"

    return numbered_text + references_section.strip()
