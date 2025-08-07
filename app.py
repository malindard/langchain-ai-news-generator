import streamlit as st
from langchain_utils.llm_selector import get_llm
from langchain_utils.memory import store_fact
from langchain_utils.research_chain import run_research
from langchain_utils.validate_chain import run_validation
from langchain_utils.writer_chain import run_writer
from langchain_utils.citation_chain import run_citation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from docx import Document
import tempfile
import os
import random

# ----------------------------
# File generation functions & Initialization
# ----------------------------

def save_as_pdf(text):
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    doc = SimpleDocTemplate(temp_path)
    styles = getSampleStyleSheet()
    story = [Spacer(1, 12)] + [Paragraph(p, styles["Normal"]) for p in text.split('\n') if p.strip()]
    doc.build(story)
    return temp_path

def save_as_docx(text):
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".docx").name
    doc = Document()
    for line in text.split('\n'):
        doc.add_paragraph(line)
    doc.save(temp_path)
    return temp_path

fun_facts = [
    "🎶 aespa is coming back this September 5th — mark your calendar!",
    "🧛‍♀️ Wednesday Season 2 is out now (sorry, I'm a fan).",
    "🎤 Katseye is the next big thing? Stay tuned.",
    "💡 Pro tip: Lower LLM temperature = more serious tone.",
    #"🧠 Did you know? Qdrant memory prevents hallucination.",
    "☕️ Grab a coffee... the agents are doing their best!",
    "📡 Research agents are surfing the web for the truth..."
]

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="AI Blog Generator", layout="wide")
st.title("📚 AI News Generator, powered by LangChain")
st.markdown("Build factual, validated, and cited blogs using multiple agents.")

# Input UI
topic = st.text_input("Enter your topic", placeholder="e.g., Latest trends in GenAI")
llm_choice = st.selectbox(
    "Choose LLM", 
    ["Mistral Small 3.1", "Gemma 3"]
)
temperature = st.slider(
    "LLM Temperature", 0.0, 1.0, 0.15,
    help="Lower = more factual and precise. Higher = more creative and varied, but possibly less accurate."
)
st.caption("⚠️ Heads up! This AI may struggle with very recent news or highly technical subjects. Please review the results carefully.")

if st.button("🚀 Generate News"):
    if not topic.strip():
        st.warning("Please enter a topic first.")
        st.stop()

    st.info("Start generating news...")
    llm = get_llm(llm_choice, temperature)

    with st.spinner(random.choice(fun_facts)):
        facts, raw_research = run_research(topic, llm)

    with st.spinner(random.choice(fun_facts)):
        validated_facts = run_validation(topic, raw_research, llm)

    # Store validated facts in Qdrant
    for f in validated_facts:
        store_fact(f["fact"], {"source": f["source"]})

    with st.spinner(random.choice(fun_facts)):
        draft_blog = run_writer(topic, validated_facts, llm)

    with st.spinner(random.choice(fun_facts)):
        final_blog = run_citation(draft_blog, llm)

    # Output
    st.subheader("📝 Final Blog")
    st.markdown(final_blog)

    pdf_path = save_as_pdf(final_blog)
    docx_path = save_as_docx(final_blog)

    st.download_button("📥 Download as PDF", data=open(pdf_path, "rb"), file_name=f"{topic}.pdf")
    st.download_button("📥 Download as DOCX", data=open(docx_path, "rb"), file_name=f"{topic}.docx")

st.markdown("---")
st.caption("Built with LangChain + Tavily + Wikipedia + Qdrant + OpenRouter")