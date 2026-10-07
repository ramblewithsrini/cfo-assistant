"""Split each policy into one chunk per section, so every answer can cite a section."""
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: str  # e.g. "procurement#3"
    policy: str  # e.g. "Procurement Policy"
    section: str  # e.g. "3. Approval limits"
    text: str

    @property
    def citation(self) -> str:
        return f"{self.policy}, section {self.section.split('.')[0]}"


def chunk_policy(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    title = re.search(r"^# (.+)$", raw, re.M).group(1).strip()
    chunks = []
    for match in re.finditer(r"^## (.+?)\n(.*?)(?=^## |\Z)", raw, re.M | re.S):
        heading, body = match.group(1).strip(), match.group(2).strip()
        number = heading.split(".")[0]
        chunks.append(
            Chunk(
                id=f"{path.stem}#{number}",
                policy=title,
                section=heading,
                # The title and heading travel with the text: they help retrieval find the right section.
                text=f"{title} — {heading}\n{body}",
            )
        )
    return chunks


def load_chunks(folder: Path) -> list[Chunk]:
    return [c for p in sorted(folder.glob("*.md")) for c in chunk_policy(p)]
