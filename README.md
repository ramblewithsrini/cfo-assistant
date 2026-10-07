# CFO-office policy assistant

A retrieval-augmented generation (RAG) assistant that answers finance-policy questions
for a **fictional** company, cites the policy section behind every answer, and refuses
rather than guesses when the policies don't cover a question.

Built to show, end to end, how I'd take an applied-AI use case from idea to a measured
proof of concept: retrieval, grounded generation, guardrails and evaluation.

## How it works

```
question ─► guardrail ─► retrieve ─► decide ─► generate ─► check ─► answer + sources
            (personal    (top 3      (below      (Claude,     (redact
             data,        policy      threshold:  answers      personal
             predictions) sections)   refuse)     only from    data)
                                                  passages)
```

| Step | Practice mode (no AWS) | Bedrock mode |
| --- | --- | --- |
| Embeddings | TF-IDF word vectors | Amazon Titan Text Embeddings V2 (matches by meaning) |
| Vector search | In-memory cosine similarity | Same; at scale, a vector database |
| Answer | Extractive: best sentences, with citations | Claude on Amazon Bedrock, grounded in the passages |
| Guardrails | Local rules | Local rules, plus Amazon Bedrock Guardrails if configured |

## Run it

```bash
python -m pip install -r requirements.txt
python ask.py "Who approves a £12,000 purchase?"
python evaluate.py
python -m pytest
python -m streamlit run app.py
```

## Switch to Amazon Bedrock

1. In the Bedrock console (for example Ireland, eu-west-1), enable **Anthropic Claude** and
   **Amazon Titan Text Embeddings V2**.
2. Sign in from the command line with temporary credentials, never long-lived keys:
   `aws login --profile bedrock` (uses your console sign-in; needs `botocore[crt]`) or
   `aws configure sso`.
3. Copy the model IDs from the console, then (PowerShell):

```powershell
$env:AWS_PROFILE = "bedrock"
$env:RAG_MODE = "bedrock"
$env:AWS_REGION = "eu-west-1"
$env:CHAT_MODEL_ID = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
python evaluate.py
```

New accounts have low request quotas, so the clients retry with adaptive back-off and
`evaluate.py` pauses between questions on Bedrock (`EVAL_DELAY`, default 3 seconds).

If antivirus software inspects HTTPS traffic (Norton, for example), the AWS tools reject its
certificate. Point them at a bundle that includes the antivirus root certificate with
`ca_bundle` in `~/.aws/config` or the `AWS_CA_BUNDLE` environment variable.

Optional: create a guardrail in Bedrock and set `GUARDRAIL_ID` and `GUARDRAIL_VERSION`.

## Results

| Measure | Practice mode | Bedrock mode (Titan V2 + Claude Haiku 4.5, eu-west-1) |
| --- | --- | --- |
| Right policy section retrieved | 20/20 | 20/20 |
| Answer contains the expected fact | 18/20 | 19/20 |
| Out-of-scope questions refused | 3/3 | 3/3 |

Bedrock run: 7 October 2026, temperature 0. Full output in `eval-bedrock-output.txt`.

Practice mode's two misses are instructive: word matching can't reason that £12,000 is
over the £10,000 limit, and can't tell that a taxi is an expense. That is exactly what
embeddings and an LLM add: on Bedrock, Claude answered both correctly.

Bedrock's one "miss" is a scoring limit, not a wrong answer. Asked whether an order can be
split to avoid CFO approval, Claude answered "No, you cannot split an order... related orders
to the same supplier within 30 days are treated as one purchase", which is correct, but the
test looks for the exact policy wording "must not be split". Exact-phrase checks are cheap
and strict; the next step would be an LLM-as-judge check for meaning, with a human reviewing
a sample of its verdicts.

## Design choices

- **Cite everything.** Each answer names its policy section, so a finance team can check it.
- **Refuse rather than guess.** Below a similarity threshold, the assistant says it can't find
  the answer. A confident wrong answer about approval limits is worse than no answer.
- **Measure before scaling.** A fixed test set scores retrieval, answers and refusals, so
  changes to prompts, models or chunking are judged on evidence.
- **Cheap by design.** A small in-memory index instead of a managed vector database: the
  whole proof of concept costs pennies to run.

All policies and people are fictional.
