"""Write the answer from the retrieved passages only.

LocalGenerator: picks the most relevant sentences (extractive), so practice mode needs no model.
BedrockGenerator: Claude on Bedrock, told to answer only from the passages and cite them.
"""
import re

from .embed import tokens

SYSTEM_PROMPT = """You are the finance policy assistant for Aldermoor Group.
Answer ONLY from the policy passages provided. Do not use outside knowledge.
After each fact, cite its source in square brackets exactly as given, e.g. [Procurement Policy, section 3].
If the passages do not contain the answer, reply exactly: "I can't find that in our finance policies."
Keep answers short and plain: two or three sentences."""

NOT_FOUND = "I can't find that in our finance policies."


def build_prompt(question: str, passages) -> str:
    context = "\n\n".join(f"[{c.citation}]\n{c.text}" for c, _ in passages)
    return f"Policy passages:\n\n{context}\n\nQuestion: {question}"


class LocalGenerator:
    def answer(self, question: str, passages) -> str:
        q = set(tokens(question))
        best = []
        for chunk, _ in passages:
            body = chunk.text.split("\n", 1)[1]
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", body):
                overlap = len(q & set(tokens(sentence)))
                if overlap:
                    best.append((overlap, sentence.strip(), chunk.citation))
        if not best:
            return NOT_FOUND
        best.sort(key=lambda x: x[0], reverse=True)
        top = best[:2]
        return " ".join(f"{s} [{cite}]" for _, s, cite in top)


class BedrockGenerator:
    def __init__(self, region: str, model_id: str, guardrail_id: str = "", guardrail_version: str = "DRAFT"):
        import boto3
        from botocore.config import Config

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            # New accounts have low request quotas: back off and retry instead of failing.
            config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
        )
        self.model_id = model_id
        self.guardrail = (
            {"guardrailIdentifier": guardrail_id, "guardrailVersion": guardrail_version} if guardrail_id else None
        )

    def answer(self, question: str, passages) -> str:
        kwargs = {
            "modelId": self.model_id,
            "system": [{"text": SYSTEM_PROMPT}],
            "messages": [{"role": "user", "content": [{"text": build_prompt(question, passages)}]}],
            "inferenceConfig": {"maxTokens": 400, "temperature": 0},
        }
        if self.guardrail:
            kwargs["guardrailConfig"] = self.guardrail
        response = self.client.converse(**kwargs)
        return response["output"]["message"]["content"][0]["text"].strip()
