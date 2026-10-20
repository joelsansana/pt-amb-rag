# pt-amb-rag

> **An eval-gated RAG system over Portuguese environmental legislation.**
> A fork of [`eu-ai-rag`](https://github.com/joelsansana/eu-ai-rag) — the
> same engineering discipline (golden set → metrics → CI regression
> gate), a different country and a different corpus.

| | |
|---|---|
| **Status** | In active development — Phases 0–1 complete (fork, DRE ingestion) |
| **License** | [MIT](./LICENSE) |
| **Upstream** | [`joelsansana/eu-ai-rag`](https://github.com/joelsansana/eu-ai-rag) |
| **Corpus** | Diário da República — APA-administered instruments (REI, Seveso III, RGGR, AIA, waters) |
| **Language** | pt-PT |

---

## What this is

A compliance Q&A system over Portuguese environmental law: it answers
questions like *"quais são os deveres gerais do operador no âmbito do
regime de prevenção de acidentes graves?"* and cites the exact
`Decreto-Lei`, `artigo`, `n.º` and `alínea` it relied on.

The interesting part is not the retrieval — it is the **eval loop**: a
golden set, Hit@5 / MRR@10 / Recall@10 metrics, an LLM-as-judge
faithfulness check, and a CI gate that fails a pull request when
retrieval quality drops. That loop is inherited from the upstream EU
project and is the reason this fork exists rather than a wrapper
around a vector store.

## How it differs from the upstream project

| | `eu-ai-rag` | `pt-amb-rag` |
|---|---|---|
| Corpus | EU AI Act + NIS2 (EUR-Lex XHTML) | Portuguese diplomas (DRE screenservice API) |
| Language | English | Portuguese — prompts, judge rubric, citation grammar, chunk headers |
| Embedder | `BAAI/bge-small-en-v1.5` (384d, English-only) | `BAAI/bge-m3` (1024d, multilingual) |
| Citations | `[Regulation X, Article Y]` | `[Decreto-Lei n.º 150/2015, art. 8.º, n.º 2, al. b)]` |
| Corpus config | Hardcoded CELEX ids | `corpora.yaml` manifest |

See [`UPSTREAM.md`](./UPSTREAM.md) for the full cherry-pick contract:
which files port cleanly from upstream, which two are permanent
conflict zones, and which must change atomically.

## Getting started

```bash
git clone git@github.com:joelsansana/pt-amb-rag.git
cd pt-amb-rag
uv sync --all-extras

# 1. Fetch the corpus declared in corpora.yaml from the Diário da República
uv run python scripts/fetch_corpus.py

# 2. Start Qdrant and build the index
docker compose up -d qdrant
uv run python scripts/build_index.py

# 3. Ask a question
uv run python -c "
from pt_amb_rag.api.ask import ask
print(ask('Quais são os deveres gerais do operador?')['answer'])
"
```

The corpus is not committed: `data/` is gitignored and rebuilt from
source, exactly as upstream does with EUR-Lex. The manifest pins every
diploma, and `data/raw/<id>.sha256` records the hash of what was
fetched so that drift is detected rather than silently absorbed.

### Running the API

```bash
docker compose up            # app + qdrant
curl -H "Authorization: Bearer $API_TOKEN" \
     -d '{"question": "O que é um acidente grave?"}' \
     http://localhost:8000/ask
```

## How the corpus is fetched

The Diário da República is a JavaScript single-page app: every URL
returns the same 2 KB shell. `src/pt_amb_rag/ingestion/dre.py` speaks
the same screenservice API the official site uses, with no browser and
no scraping, and parses DRE's own `TextoFormatado` markup into
structural units (`Artigo`, `n.º`, `alínea`, `Anexo`, `CAPÍTULO`).

The details — including three transport quirks that are load-bearing
and two open legal questions — are documented in
[`PILOT_FINDINGS.md`](./PILOT_FINDINGS.md).

## Repository layout

```text
pt-amb-rag/
├── corpora.yaml                  # the corpus manifest — add a diploma here
├── UPSTREAM.md                   # cherry-pick contract with eu-ai-rag
├── PILOT_FINDINGS.md             # what the DRE spike found
├── src/pt_amb_rag/
│   ├── config.py                 # corpus + language + embedder configuration
│   ├── ingestion/
│   │   ├── dre.py                # DRE client + structural parser
│   │   └── chunker.py            # PT contextual headers
│   ├── retrieval/                # embedder + Qdrant vector store
│   ├── generation/               # LLM client + PT prompt
│   ├── api/                      # FastAPI app
│   └── eval/                     # metrics, judge, golden generator, corpus pins
├── scripts/
│   ├── fetch_corpus.py           # manifest-driven DRE fetcher
│   ├── build_index.py            # chunks → Qdrant
│   ├── build_golden.py           # golden-set candidate generation
│   ├── vet_golden.py             # interactive curation
│   ├── run_eval.py               # retrieval metrics
│   ├── run_eval_gen.py           # LLM-as-judge
│   └── eval_gate.py              # CI regression gate
├── evals/golden/                 # golden sets (tiers: principal, unanswerable)
└── tests/                        # unit + integration
```

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright src tests
uv run pytest -q
```

All four run in CI on every push and pull request.

## Planning documents

| Document | What it is |
|---|---|
| [`Fork Engineering Plan.md`](./Fork%20Engineering%20Plan.md) | The plan this fork executes — repo strategy, keep/parameterize/rewrite/drop decisions, phase gates |
| [`RAG Methodology Upgrade Plan.md`](./RAG%20Methodology%20Upgrade%20Plan.md) | The retrieval-quality programme — embedder, hybrid search, chunking, eval statistics |
| [`PILOT_FINDINGS.md`](./PILOT_FINDINGS.md) | What the corpus spike found, and where the plan was wrong |
| [`legal-citation-ledger.json`](./legal-citation-ledger.json) | Verified DRE references for the instruments in scope |

## Status

Phases 0–1 are complete: the fork is renamed and green in CI, the
corpus is manifest-driven, and DRE ingestion works end to end for ten
diplomas. The golden set — the centrepiece of the design — is the next
piece of work. See
[`PILOT_FINDINGS.md` §9](./PILOT_FINDINGS.md#9-recommended-next-steps-in-order)
for the ordered list.
