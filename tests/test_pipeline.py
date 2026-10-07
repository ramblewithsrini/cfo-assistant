"""Practice-mode tests: run with  python -m pytest"""
from rag.chunk import load_chunks
from rag.config import settings
from rag.guard import check_input
from rag.pipeline import Assistant

BOT = Assistant()


def test_every_policy_is_split_into_sections():
    chunks = load_chunks(settings.policies_dir)
    assert len({c.id.split("#")[0] for c in chunks}) == 12
    assert all(c.text.startswith(c.policy) for c in chunks)


def test_answers_cite_the_right_policy():
    r = BOT.ask("Who approves a purchase over £10,000?")
    assert "procurement#3" in [s[1] for s in r.sources]
    assert "Procurement Policy, section 3" in r.answer


def test_personal_data_requests_are_blocked():
    allowed, reason = check_input("What is the salary of Priya?")
    assert not allowed and "personal" in reason


def test_unrelated_questions_are_refused_not_guessed():
    r = BOT.ask("What is the capital of France?")
    assert not r.grounded
    assert "can't find" in r.answer
