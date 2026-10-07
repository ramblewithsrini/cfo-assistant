"""The whole RAG flow: check, retrieve, decide, generate, check again."""
from dataclasses import dataclass, field

from .config import settings
from .embed import BedrockEmbedder, LocalEmbedder, tokens
from .generate import NOT_FOUND, BedrockGenerator, LocalGenerator
from .guard import check_input, redact_output
from .index import VectorIndex


@dataclass
class Result:
    question: str
    answer: str
    sources: list = field(default_factory=list)  # (citation, chunk id, score)
    guardrail: str = "Passed"
    grounded: bool = True


def make_components(s=settings):
    if s.mode == "bedrock":
        embedder = BedrockEmbedder(s.region, s.embed_model_id)
        generator = BedrockGenerator(s.region, s.chat_model_id, s.guardrail_id, s.guardrail_version)
    else:
        embedder, generator = LocalEmbedder(), LocalGenerator()
    return VectorIndex(embedder), generator


class Assistant:
    def __init__(self, s=settings):
        self.s = s
        self.index, self.generator = make_components(s)
        if s.index_path.exists():
            self.index.load(s.index_path)
        else:
            self.index.build(s.policies_dir)
            self.index.save(s.index_path)

    def ask(self, question: str) -> Result:
        allowed, reason = check_input(question)
        if not allowed:
            return Result(question, "I can't help with that request.", guardrail=reason, grounded=False)

        hits = self.index.search(question, self.s.top_k)
        sources = [(c.citation, c.id, round(score, 3)) for c, score in hits]
        relevant = [(c, score) for c, score in hits if score >= self.s.min_score]
        if self.s.mode == "local":
            # Word matching can't tell 'capital of France' from 'capital expenditure', so practice
            # mode also needs two shared words. Bedrock embeddings judge by meaning instead.
            q = set(tokens(question))
            need = min(2, len(q))
            relevant = [(c, s) for c, s in relevant if len(q & set(tokens(c.text))) >= need]
        if not relevant:
            # Nothing close enough: refuse rather than let the model guess.
            return Result(question, NOT_FOUND, sources, "Passed: no relevant policy found", grounded=False)

        answer = redact_output(self.generator.answer(question, relevant))
        return Result(question, answer, sources, reason, grounded=answer != NOT_FOUND)
