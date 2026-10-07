"""Score the assistant against eval/questions.json.

Three measures, the kind a client would sign off a proof of concept on:
  retrieval  - was the right policy section among the sources?
  answer     - does the answer contain the expected fact?
  refusal    - did out-of-scope questions get refused instead of answered?
"""
import json
import os
import sys
import time
from pathlib import Path

from rag.pipeline import Assistant

ROOT = Path(__file__).parent


def main() -> int:
    questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    bot = Assistant()
    in_scope = [q for q in questions if q["source"]]
    out_scope = [q for q in questions if not q["source"]]
    retrieval = answer = refusal = 0

    # Pause between calls on Bedrock so a new account's request quota isn't exceeded.
    delay = float(os.getenv("EVAL_DELAY", "3" if os.getenv("RAG_MODE") == "bedrock" else "0"))
    for q in questions:
        time.sleep(delay)
        r = bot.ask(q["q"])
        ids = [s[1] for s in r.sources]
        hit = q["source"] in ids if q["source"] else None
        ok = all(e.lower() in r.answer.lower() for e in q["expect"])
        if q["source"]:
            retrieval += hit
            answer += ok
        else:
            refusal += ok
        mark = "PASS" if ok and hit is not False else "FAIL"
        print(f"{mark}  {q['q']}\n      -> {r.answer[:150]}")
        if hit is False:
            print(f"      expected source {q['source']}, got {ids}")

    print("\n--- Scores ---")
    print(f"Retrieval (right section found): {retrieval}/{len(in_scope)}")
    print(f"Answer contains expected fact:   {answer}/{len(in_scope)}")
    print(f"Out-of-scope correctly refused:  {refusal}/{len(out_scope)}")
    return 0 if retrieval == len(in_scope) and refusal == len(out_scope) else 1


if __name__ == "__main__":
    sys.exit(main())
