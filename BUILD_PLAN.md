# Build Plan — `pt-amb-rag`

> **A pre-registered, eval-gated RAG system over Portuguese
> environmental legislation.** This document is the fork's build plan:
> what was built, in what order, and what changed along the way.
>
> The engineering decisions live in
> [`Fork Engineering Plan.md`](./Fork%20Engineering%20Plan.md); the
> retrieval-quality programme lives in
> [`RAG Methodology Upgrade Plan.md`](./RAG%20Methodology%20Upgrade%20Plan.md);
> the verified findings from the corpus spike live in
> [`PILOT_FINDINGS.md`](./PILOT_FINDINGS.md).

| | |
|---|---|
| **Status** | In active development — Phases 0–1 complete |
| **License** | [MIT](./LICENSE) |
| **Upstream** | [`joelsansana/eu-ai-rag`](https://github.com/joelsansana/eu-ai-rag) |
| **Author** | Joel Sansana |

---

## Why this project exists

The upstream `eu-ai-rag` exists for two reasons at once: a Lepanto
sales tool that answers EU-regulation questions with citations, and a
portfolio piece with a defensible engineering story. The same two
reasons hold here, with the same emphasis on the second one.

1. **A compliance tool for Portuguese chemical and industrial
   operators.** The instruments that matter for this audience — the
   licensing regime (REI, DL 127/2013), major-accident prevention
   (Seveso III, DL 150/2015), waste management (RGGR, DL 102-D/2020),
   environmental impact assessment (AIA, DL 151-B/2013), and water
   resources — are exactly the corpus in `corpora.yaml`. A RAG over
   them can be handed to a customer during a sales conversation and
   asked *"what does the operator owe us here, and with which
   citation?"*
2. **A portfolio piece with a defensible engineering story.** The
   interesting part is not the retrieval — it is the **eval loop**: a
   golden set, Hit@5 / MRR metrics, an LLM-as-judge faithfulness
   check, and a CI regression gate that fails a pull request when
   retrieval quality drops. That is the pattern every serious ML
   system needs, and it is inherited from upstream rather than
   reinvented.

---

## The build, in one table

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Fork, rename to `pt_amb_rag`, config + manifest layer, CI green | ✅ Done |
| 1 | DRE ingestion: client, structural parser, PT chunk grammar, manifest-driven fetcher | ✅ Done |
| 2 | Retrieval v1: multilingual embedder from config, index build, PT answers | ⬜ Next |
| 3 | API + Docker port | ⬜ Next |
| 4 | **Eval loop** — golden sets, metrics, judge, CI gate | ⬜ Planned |
| 5 | Hybrid retrieval (BM25-PT + dense + RRF) and/or reranker | ⬜ Conditional |
| 6 | PDF lane for the numeric tables, consolidated text, README/launch | ⬜ Planned |

**The order is deliberate.** Phase 4 (evaluation) lands *before* any
retrieval upgrade, so that every improvement is quantified against a
measured baseline rather than asserted. Phase 5 is conditional: if the
baseline already meets the pre-registered threshold, Phase 5 is not an
upgrade, it is a story.

---

## Architecture

```text
┌─────────────────────┐    ┌────────────────────────┐    ┌─────────────────────┐
│  Diário da República │    │  Ingestion pipeline    │    │  Qdrant vector store │
│  (screenservice API)│───▶│  - DRE client          │───▶│  - Article-level    │
│  DL 127/2013 …      │    │  - structural parser   │    │    chunking         │
│                     │    │  - PT contextual hdrs  │    │  - payload metadata │
│                     │    │  - content-hash version│    │  (reg, art, n.º, …) │
└─────────────────────┘    └────────────────────────┘    └─────────────────────┘
                                                                   │
                                                                   ▼
┌─────────────────────┐    ┌────────────────────────┐    ┌─────────────────────┐
│  FastAPI app        │◀───│  Retrieval layer       │◀───│  Embedder           │
│  POST /search       │    │  - dense (bge-m3)      │    │  BAAI/bge-m3        │
│  POST /ask          │    │  - hybrid (planned:    │    │  (local, 1024-d)    │
│  GET  /health       │    │    BM25-PT + dense+RRF)│    │                     │
└─────────────────────┘    └────────────────────────┘    └─────────────────────┘
             │
             ▼
┌─────────────────────┐
│  Generation layer   │
│  - MiniMax (OpenAI- │
│    compatible API)  │
│  - PT citation      │
│    grammar          │
│  - PT abstention    │
└─────────────────────┘
```

---

## Decisions (defaults, change with reason)

| Decision | Choice | Why |
|---|---|---|
| **Corpus** | Portuguese diplomas from the DRE, declared in `corpora.yaml` | APA-administered instruments are the audience's actual exposure |
| **Transport** | The DRE screenservice API, not scraping | The DRE serves no document HTML; the API returns the structured document |
| **Embeddings** | `BAAI/bge-m3` (local, 1024-d) | The upstream English-only encoder never saw Portuguese; bge-m3 has the best published PT evidence and an MIT licence |
| **Vector store** | Qdrant via docker-compose | Payload filters make article-level metadata scoping trivial |
| **LLM** | MiniMax via the OpenAI-compatible API | Same client shape upstream uses; ~€5 per dev cycle |
| **Frameworks** | None. Hand-rolled | Framework wrappers signal tutorial-following; hand-rolled signals understanding |
| **Package mgmt** | `uv`, Python 3.12, `ruff` + `pyright` + `pytest` | Inherited from upstream; it works |
| **Pre-registered threshold** | Phase 4 must hit ≥0.85 Hit@5 before Phase 5 is justified | Makes the v1→v2 narrative defensible whichever way the numbers go |

---

## What this build demonstrates

| Capability | Where it lives |
|---|---|
| **Eval-gated ML.** Golden set, retrieval metrics, LLM-as-judge, CI regression gating | `src/pt_amb_rag/eval/`, `scripts/run_eval*.py`, `scripts/eval_gate.py` |
| **Structured-corpus retrieval.** Article-aware chunking, PT contextual headers, payload filters | `src/pt_amb_rag/ingestion/`, `src/pt_amb_rag/retrieval/` |
| **Citations as a faithfulness mechanism.** Every answer links to a diploma, article, número and alínea | `src/pt_amb_rag/generation/prompts.py` |
| **Honest abstention.** Unanswerable questions return a deterministic Portuguese sentinel rather than a hallucinated attempt | `corpora.yaml` → `language.abstention`, `eval/judge.py` |
| **Reverse-engineered, documented source access.** An undocumented API, reverse-engineered from the deployed bundle, with the contract written down | `src/pt_amb_rag/ingestion/dre.py`, `PILOT_FINDINGS.md` |
| **Production-shape packaging.** `pyproject.toml`, multi-stage Docker, slim base, non-root user | `Dockerfile`, `docker-compose.yml` |

---

## Known limitations

Populated as limits are discovered, not at the end.

- **Numeric tables are not in the corpus yet.** DRE's HTML carries no
  table markup; the emission-limit values live only in the official
  PDF. A corpus built from `TextoFormatado` alone cannot answer
  emission-limit questions — the questions this audience asks most.
  See [`PILOT_FINDINGS.md` §4](./PILOT_FINDINGS.md).
- **Only the original text is ingested.** The consolidated version is a
  different screen; it can be pinned to a specific consolidation date,
  but that is not implemented. Until it is, the corpus answers "what
  the text said when published", not "what the law says today".
- **Consolidated text has no legal value.** DRE's consolidated editions
  are marked *"texto consolidado sem valor legal"*. When the
  consolidated lane lands, every answer derived from it must disclose
  that.
- **The screenservice API is undocumented and could move.** A DRE
  deploy could change it. The client reads the live bundle for its
  tokens; a smoke test should fail loudly when the contract changes.
- **Redistribution terms are unconfirmed.** The DRE's open-data regime
  and the INCM-produced PDFs have not been reviewed by a lawyer.
- **The gate is statistically underpowered.** The upstream golden set
  (79 items) cannot distinguish 0.806 from 0.87 — the CI band is
  ±0.09. This fork inherits the same limit; the fix is a larger
  golden set and a paired significance test, per the methodology plan.

---

## How to read this project

- **You want to run it:** [`README.md`](./README.md) § *Getting started*.
- **You want to extend the corpus:** add an entry to `corpora.yaml`.
  No code change.
- **You want to understand the DRE integration:** read
  [`PILOT_FINDINGS.md` §1](./PILOT_FINDINGS.md) and
  `src/pt_amb_rag/ingestion/dre.py`.
- **You want to sync with upstream:** read
  [`UPSTREAM.md`](./UPSTREAM.md).
- **You want to evaluate the engineering:** the test suite under
  `tests/`, the parser in `src/pt_amb_rag/ingestion/dre.py`, and the
  chunker in `src/pt_amb_rag/ingestion/chunker.py`.
