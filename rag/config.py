"""Settings, read from environment variables so no keys or IDs live in the code.

RAG_MODE=local    practice mode: no AWS, no cost (default)
RAG_MODE=bedrock  Titan embeddings + Claude on Amazon Bedrock (+ optional Guardrails)
"""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    mode: str = os.getenv("RAG_MODE", "local")
    region: str = os.getenv("AWS_REGION", "eu-west-1")
    # Copy the exact IDs from the Bedrock console; inference-profile IDs start with "eu." in Europe.
    chat_model_id: str = os.getenv("CHAT_MODEL_ID", "eu.anthropic.claude-haiku-4-5-20251001-v1:0")
    embed_model_id: str = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
    guardrail_id: str = os.getenv("GUARDRAIL_ID", "")  # optional
    guardrail_version: str = os.getenv("GUARDRAIL_VERSION", "DRAFT")
    top_k: int = int(os.getenv("TOP_K", "3"))
    # Below this similarity the assistant says it can't find an answer instead of guessing.
    min_score: float = float(os.getenv("MIN_SCORE", "0.12" if os.getenv("RAG_MODE", "local") == "local" else "0.35"))
    policies_dir: Path = ROOT / "policies"
    index_path: Path = ROOT / ("index-local.json" if os.getenv("RAG_MODE", "local") == "local" else "index-bedrock.json")


settings = Settings()
