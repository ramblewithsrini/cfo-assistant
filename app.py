"""Demo screen:  streamlit run app.py

Shows the answer, the sources it came from and their similarity scores, and the
guardrail decision, so you can explain every step of RAG in an interview.
"""
import streamlit as st

from rag.config import settings
from rag.pipeline import Assistant

st.set_page_config(page_title="CFO-office assistant", page_icon="📊", layout="wide")


@st.cache_resource
def assistant():
    return Assistant()


st.title("CFO-office policy assistant")
st.caption(
    f"Retrieval-augmented generation over 12 fictional finance policies · mode: **{settings.mode}**"
    + (f" · model: {settings.chat_model_id}" if settings.mode == "bedrock" else "")
)

examples = [
    "Can I approve a £12,000 software purchase myself?",
    "A supplier emailed new bank details. What must we do before paying them?",
    "How should one-off set-up fees be recognised as revenue?",
    "What is John Smith's salary?",
]
choice = st.selectbox("Try an example, or type your own below", [""] + examples)
question = st.text_input("Your question", value=choice)

if question:
    r = assistant().ask(question)
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Answer")
        (st.success if r.grounded else st.warning)(r.answer)
    with right:
        st.subheader("How it was answered")
        st.markdown(f"**1. Guardrail on the question:** {r.guardrail}")
        st.markdown("**2. Retrieved policy sections** (similarity score):")
        for citation, cid, score in r.sources:
            used = "✅" if score >= settings.min_score else "—"
            st.markdown(f"{used} `{score:.3f}` {citation}")
        st.markdown(
            f"**3. Decision:** sections scoring below {settings.min_score} are ignored; "
            "if none qualify, the assistant refuses rather than guessing."
        )
