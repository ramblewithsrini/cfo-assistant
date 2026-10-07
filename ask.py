"""Ask one question from the command line:  python ask.py "Who approves a £5,000 purchase?" """
import sys

from rag.config import settings
from rag.pipeline import Assistant

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Who approves a £5,000 purchase?"
    r = Assistant().ask(question)
    print(f"Mode: {settings.mode}\n\nQ: {r.question}\nA: {r.answer}\n")
    print(f"Guardrail: {r.guardrail}")
    for citation, cid, score in r.sources:
        print(f"  source {score:.3f}  {citation}  ({cid})")
