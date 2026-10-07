"""Turn text into vectors.

LocalEmbedder: a small bag-of-words model so the whole pipeline runs without AWS.
BedrockEmbedder: Amazon Titan Text Embeddings V2, which matches by meaning, not just words.
"""
import json
import math
import re
from collections import Counter

STOP = set(
    "a an the and or of to in on for by with is are be can do does i we you my our it this that "
    "what when who how much many which need needs there their from at as if any not up over under".split()
)


def _stem(word: str) -> str:
    for suffix in ("ations", "ation", "ing", "ies", "ed", "es", "al", "s", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9£]+", text.lower())
    return [_stem(w) for w in words if w not in STOP]


class LocalEmbedder:
    """TF-IDF vectors over the policy vocabulary, normalised for cosine similarity."""

    def __init__(self):
        self.idf: dict[str, float] = {}

    def fit(self, texts: list[str]) -> None:
        docs = [set(tokens(t)) for t in texts]
        vocab = set().union(*docs)
        n = len(texts)
        self.idf = {w: math.log((1 + n) / (1 + sum(w in d for d in docs))) + 1 for w in vocab}

    def embed(self, text: str) -> dict[str, float]:
        counts = Counter(t for t in tokens(text) if t in self.idf)
        vec = {w: c * self.idf[w] for w, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {w: v / norm for w, v in vec.items()}

    def state(self) -> dict:
        return {"idf": self.idf}

    def load_state(self, state: dict) -> None:
        self.idf = state["idf"]


class BedrockEmbedder:
    def __init__(self, region: str, model_id: str):
        import boto3
        from botocore.config import Config

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            # New accounts have low request quotas: back off and retry instead of failing.
            config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
        )
        self.model_id = model_id

    def fit(self, texts: list[str]) -> None:  # nothing to learn: the model is pre-trained
        pass

    def embed(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text, "dimensions": 512, "normalize": True})
        response = self.client.invoke_model(modelId=self.model_id, body=body)
        return json.loads(response["body"].read())["embedding"]

    def state(self) -> dict:
        return {}

    def load_state(self, state: dict) -> None:
        pass


def cosine(a, b) -> float:
    """Both vector types are normalised, so the dot product is the cosine similarity."""
    if isinstance(a, dict):
        return sum(v * b.get(k, 0.0) for k, v in a.items())
    return sum(x * y for x, y in zip(a, b))
