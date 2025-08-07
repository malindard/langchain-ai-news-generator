import os
import re
from langchain.tools.tavily_search import TavilySearchResults
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def get_tavily_search():
    return TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))

# The prompt strictly enforces citations
RESEARCH_PROMPT = """
You are an senior research assistant with high standards for factual accuracy.
Your job is to collect ONLY objective, verifiable information about the topic below, using the web search results provided.

TOPIC: {topic}

Your constraints:
- Use ONLY facts that appear in the search results below. If a fact is NOT present, do NOT include it.
- After EVERY fact, include a markdown link to the exact source: [Source](URL)
- Do NOT include any fact, data, or name that isn’t directly supported by a cited source.
- NO speculation, NO personal opinions, NO summaries of opinions.
- If no reliable sources exist for a detail, SKIP it entirely.
- Group facts into clearly labeled sections using markdown headings (##).

FORMAT:
- Write as clean markdown with headings and bullet points.
- Each bullet point must have a [Source](URL) at the end.
- Do NOT copy web search summaries verbatim; paraphrase concisely.
- Do NOT hallucinate or invent any information.
- If the topic is too new or obscure and you can’t find real facts, reply: “Insufficient reliable information available.”

BEGIN. Use only the context provided below:

{search_results}

"""

def run_research(topic, llm):
    search_tool = get_tavily_search()
    search_results = search_tool.run(topic)
    prompt = PromptTemplate.from_template(RESEARCH_PROMPT)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Use search results as additional context to the LLM
    output = llm_chain.run(topic=topic, search_results=search_results)
    
    # Parse output into a list of (fact, url) tuples
    pattern = r"- (.*?)(?:\s*\[Source\]\((.*?)\))"
    facts = re.findall(pattern, output)
    return facts, output
