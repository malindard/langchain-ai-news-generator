import re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.utilities.wikipedia import WikipediaAPIWrapper
from langchain_utils.memory import retrieve_memory_facts

VALIDATOR_PROMPT = """
You are a senior fact-checker known for precision and honesty.

Your job is to verify the factual accuracy of a draft article, using:
- Memory facts from previous research
- Wikipedia excerpts
- General knowledge

Be strict:
- If a statement is fully supported by a memory fact or Wikipedia, mark it **True**
- If it contradicts verified information, mark it **False**, and suggest a correction
- If unclear, mark it **Uncertain**, and briefly explain why
- Do NOT guess or hallucinate facts
- Always refer to the provided memory and Wikipedia content

Format yult as:
- "Original statement" — True/False/Uncertain — short reason — Correction (if False)

Example:
- "The Eiffel Tower is in Berlin." — False — Wikipedia says it's in Paris — Correction: "The Eiffel Tower is in Paris."

FACTUAL MEMORY:
{memory}

WIKIPEDIA RESULTS:
{wikipedia}

DRAFT TEXT:
{draft}
"""

def run_validation(topic: str, facts: str, llm) -> list[dict]:
    # Retrieve memory facts from vector DB
    memory_facts = retrieve_memory_facts(topic)
    memory = (
        "\n".join([f"- {fact}" for fact in memory_facts])
        if memory_facts else "No memory found for this topic."
    )

    # Fetch Wikipedia summary directly
    wikipedia_tool = WikipediaAPIWrapper()
    wikipedia_summary = wikipedia_tool.run(topic)

    # Combine facts into one block of text
    draft_text = "\n".join(facts) if isinstance(facts, list) else facts

    # Run validation LLM chain
    prompt = PromptTemplate.from_template(VALIDATOR_PROMPT)
    chain = LLMChain(llm=llm, prompt=prompt)

    result = chain.run({
        "memory": memory,
        "wikipedia": wikipedia_summary,
        "draft": draft_text
    })

    # Parse the output
    parsed = []
    pattern = r'^- "(.*?)"\s+—\s+(True|False|Uncertain)\s+—\s+(.*?)\s+(?:—\s+Correction: "(.*?)")?$'
    for line in result.splitlines():
        match = re.match(pattern, line.strip())
        if match:
            original, verdict, reason, correction = match.groups()
            parsed.append({
                "fact": correction if correction else original,
                "source": f"{verdict} - {reason}"
            })

    return parsed
