import os
from langchain_openai import ChatOpenAI

def get_llm(llm_choice: str, temperature: float):
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise ValueError("Missing OPENROUTER_API_KEY in environment")

    model_map = {
        "Mistral Small 3.1": "mistralai/mistral-small-3.1-24b-instruct:free",
        "Gemma 3": "google/gemma-3-27b-it:free",
    }

    if llm_choice not in model_map:
        raise ValueError(f"Invalid LLM model choice: {llm_choice}")

    return ChatOpenAI(
        api_key=openrouter_key,
        model=model_map[llm_choice],
        temperature=temperature,
        base_url="https://openrouter.ai/api/v1",
    )
