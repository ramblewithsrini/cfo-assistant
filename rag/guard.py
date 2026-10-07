"""Checks before and after the model.

Practice mode uses simple local rules. In Bedrock mode, Amazon Bedrock Guardrails can
replace them (topic denial, personal-data filters, grounding checks) via GUARDRAIL_ID.
"""
import re

PERSONAL_WORDS = r"(salary|salaries|pay|bonus|performance|home address|date of birth)"
# "salary of Priya", "bonus for John Smith"
PERSONAL = re.compile(PERSONAL_WORDS + r"\b.*\b(of|for)\s+[A-Z][a-z]+")
# "John Smith's salary", "Priya's bonus"
POSSESSIVE = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?['’]s\s+" + PERSONAL_WORDS + r"\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PREDICTION = re.compile(r"\b(share price|stock price|will .* be next (year|quarter))\b", re.I)


def check_input(question: str) -> tuple[bool, str]:
    """Return (allowed, reason)."""
    if PERSONAL.search(question) or POSSESSIVE.search(question):
        return False, "Blocked: asks about an individual's personal information (Acceptable Use of AI, section 3)."
    if PREDICTION.search(question):
        return False, "Blocked: asks for a prediction, which is outside finance policy."
    return True, "Passed"


def redact_output(answer: str) -> str:
    return EMAIL.sub("[email removed]", answer)
