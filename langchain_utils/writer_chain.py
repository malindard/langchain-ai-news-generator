from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

ARTICLE_PROMPT = """
You are a professional blog writer tasked with composing a comprehensive, factual, and engaging blog article.
You must use ONLY the facts provided below, which have been validated for accuracy.

Facts:
{validated_facts}

Topic: {topic}

Guidelines:
- Structure the article with clear markdown headings (##, ###)
- Use bullet points or tables where helpful to improve readability
- DO NOT include any information not found in the facts
- Every major claim MUST include a [Source](URL) citation right after it
- Do NOT include speculation, personal opinion, or invented details
- End the article with a markdown '## References' section listing all unique URLs used

Return ONLY the article in clean, ready-to-publish markdown format.
"""

def run_writer(topic, validated_facts, llm):
    # Filter out low-confidence facts
    high_conf_facts = [f for f in validated_facts if f["confidence"] != "Low"]

    # Format facts as markdown list items
    facts_md = "\n".join(
        f"- {entry['fact']} [Source]({entry['source']})" for entry in high_conf_facts
    )

    # Prompt construction
    prompt = PromptTemplate.from_template(ARTICLE_PROMPT)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    article = llm_chain.run(validated_facts=facts_md, topic=topic)

    # Extract unique sources to append to the end manually
    unique_sources = sorted(set(entry["source"] for entry in high_conf_facts))
    references_md = "\n".join(f"- {url}" for url in unique_sources)
    article += f"\n\n## References\n{references_md}"

    return article
