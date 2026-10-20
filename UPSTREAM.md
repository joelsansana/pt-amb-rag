# Upstream relationship

`pt-amb-rag` is a fork of [`joelsansana/eu-ai-rag`](https://github.com/joelsansana/eu-ai-rag)
(the EU AI Act + NIS2 RAG). This file is the contract that governs how
the two repositories stay in sync.

The fork is **not** a generic framework with a jurisdiction plug-in: the
corpus, the language, the citation grammar and the prompt set are all
different, and the divergent half is larger than the shared half. What
is shared is the *eval loop* — the golden set, the metrics, the CI
regression gate — and that part is kept deliberately close to upstream
so that upstream fixes cherry-pick cleanly.

## Portable paths — cherry-pick these freely

These files are (or should be) byte-comparable with upstream. When
upstream fixes a bug here, cherry-pick it and keep the local change
surface at zero.

| Path | Why it ports unchanged |
|---|---|
| `src/pt_amb_rag/eval/metrics.py` | Pure id arithmetic on `hit_at_k` / `mrr_at_k` / `recall_at_k`. No corpus, no language. |
| `src/pt_amb_rag/eval/sampling.py` | `content_hash::question` keys and seeded sampling. |
| `src/pt_amb_rag/eval/prefilter.py` | Length and `?`-suffix heuristics. The question mark is the same character in Portuguese. |
| `scripts/eval_gate.py` | Metric key, tolerance, baseline URL. Fully generic, including the bootstrap-pass on a missing baseline. |
| `src/pt_amb_rag/api/auth.py` | Env-var token auth. |
| `src/pt_amb_rag/api/main.py` | Route shape and the threadpool pattern. Only the FastAPI title differs. |
| `.github/workflows/ci.yml` | ruff + ruff format + pyright + pytest. Zero corpus coupling. |

## Permanent conflict zones

These two files were restructured and **will** conflict on every
upstream change. Resolve by hand, in the fork's favour.

| Path | Local divergence | Upstream divergence |
|---|---|---|
| `src/pt_amb_rag/eval/corpus_pins.py` | `check_record_pins(record, corpus_paths: dict[str, Path])` — generalised from the upstream two-file signature to the N diplomas in `corpora.yaml`. | Upstream keeps `check_record_pins(record, ai_act_path, nis2_path)`. |
| `scripts/run_eval.py` | `--tier` choices come from `corpora.yaml` (`eval_tiers`), not a hardcoded list. | Upstream hardcodes `choices=["lepanto", "demo"]`. |

## Deliberately diverged

| Path | Upstream | Fork |
|---|---|---|
| `src/pt_amb_rag/ingestion/` | `eur_lex.py` — parses EUR-Lex XHTML (`.eli-subdivision[id^='art_']`, `.oj-ti-art`, …). | `dre.py` — speaks the DRE screenservice API and parses DRE's own `TextoFormatado` markup. Different corpus, different transport, different markup. |
| `src/pt_amb_rag/ingestion/chunker.py` | EU header grammar (`Article 12, Regulation (EU) 2024/1689 (AI Act), effective …`). | PT header grammar (`Artigo 12.º — Decreto-Lei n.º 127/2013, de 30 de agosto`). |
| `src/pt_amb_rag/generation/prompts.py` | English RAG prompt, `[Regulation X, Article Y]` citations. | Portuguese RAG prompt, `[Decreto-Lei n.º X, art. Y.º, n.º Z]` citations. |
| `src/pt_amb_rag/eval/judge.py` | English rubric. | Portuguese rubric — a cross-language judge is measurably less consistent than a same-language one. |
| `src/pt_amb_rag/eval/golden_generator.py` | English prompt. | Portuguese prompt. |
| `src/pt_amb_rag/retrieval/embedder.py` | `BAAI/bge-small-en-v1.5`, 384 dims. | `BAAI/bge-m3`, 1024 dims, dimension asserted against the model metadata. |
| `src/pt_amb_rag/retrieval/vector_store.py` | `celex` / `recital_num` payload fields. | `eli` / `numero` / `alinea` / `diploma` payload fields. The mechanics (`_point_id`, `_build_filter`, `ensure_collection`) are unchanged. |
| `src/pt_amb_rag/api/schemas.py` | `celex` required on a source chunk. | `eli` required, plus the PT locator fields. |
| `corpora.yaml` | — (no manifest upstream; CELEX ids are hardcoded in `eur_lex.py` and in `eval.yml`) | The corpus manifest. Adding a diploma is a manifest edit. |
| `src/pt_amb_rag/config.py` | — | Single source of truth for corpus, language, embedder, tier list. |

## Atomicity rule

The language-dependent files move **together** or not at all:

```
src/pt_amb_rag/generation/prompts.py   ← abstention sentinel
src/pt_amb_rag/eval/judge.py           ← matches the sentinel
src/pt_amb_rag/eval/golden_generator.py
corpora.yaml                           ← language.abstention
```

Changing the abstention string in one place and not the others silently
mis-scores the unanswerable tier: the assistant abstains with one
phrase and the judge looks for another. The same applies to the
citation grammar and to the chunk header format, which the golden set's
`gold_ids` are derived from.

## When to extract a shared package

If a **third** jurisdiction fork appears, extract `legalrag-core`
(metrics + eval gate + runners + vector-store mechanics) and make both
forks depend on it. Before that, the packaging, versioning and
cross-repo sync cost more than the ~900 lines it would deduplicate.

## Decision trigger for revisiting this contract

Upstream's `run_eval.py` already carries a fix lineage for the
`k_rank` bug in its docstring. If that lineage is extended, cherry-pick
it and preserve the local `--tier` handling: keep the upstream bug fix
and the fork's config-driven tier list in the same file, and keep
`corpus_pins`'s two-file signature change in the other.
