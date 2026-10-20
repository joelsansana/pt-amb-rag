# Engineering Plan — Forking `eu-ai-rag` → `pt-amb-rag`

Portuguese environmental-regulation RAG for the chemical industry (APA-administered instruments).
Grounded in the actual repo at `~/Documents/projects/eu-ai-rag` @ `72986ab` (main, phases 0–4 + playground done).
All file references below were read; LOC figures from `wc -l`. **Analysis only — no code written, repo untouched.**

Repo stats: ~3,471 LOC across `src/safety_rag` (20 modules), `scripts/` (8 CLIs), `tests/eval/` (11 files). Golden sets: `lepanto.jsonl` (31 q), `demo.jsonl` (36), `unanswerable.jsonl` (12), candidate pools 969 + 621 lines. Baseline (committed): Hit@5 0.806, MRR@10 0.761, Recall@10 0.968. Corpus: 323 + 207 chunks.

---

## 1. Keep / Parameterize / Rewrite / Drop

### Keep as-is (language- and corpus-agnostic, verified)
| Module | Why it ports unchanged |
|---|---|
| `eval/metrics.py` (23 LOC) | Pure id-math on `hit_at_k`/`mrr_at_k`/`recall_at_k`. |
| `eval/sampling.py` (28) | `content_hash::question` keys, seeded sampling. |
| `eval/prefilter.py` (34) | Length/`?`-suffix heuristics hold for PT (`?` same char). |
| `scripts/eval_gate.py` (114) | Fully generic: metric key + tolerance + baseline URL. Bootstrap-pass on missing baseline preserved. |
| `retrieval/vector_store.py` mechanics | `_point_id` (uuid5 on content_hash → idempotent upsert), `_build_filter` (regulation/article_num/part), `ensure_collection`. Only constants change (see below). |
| `api/auth.py`, `api/main.py` route shape, `api/ask.py` | Env-token auth; threadpool pattern; `ask(answer=False)` keyless-CI path. |
| `.github/workflows/ci.yml` | ruff + format + pyright + pytest; zero corpus coupling. |
| `evals/golden/CURATION_GUIDELINES.md` | Discipline doc; translate + adapt examples (gold-chunk correctness, anti-parroting, difficulty calibration). |
| Build discipline itself | Eval loop (Phase 4) **before** retrieval upgrades (Phase 5); corpus rebuilt from source in CI; pins verify lockstep. |

### Parameterize (move hardcodes → config; keep logic)
| Axis | Today (hardcoded) | Fork target |
|---|---|---|
| Corpus source | `CELEX_TO_REGULATION` / `CELEX_TO_EFFECTIVE_DATE` dicts in `eur_lex.py`; `CELLAR_URL` in `download_corpus.py` | `corpora.yaml` manifest: diploma id, source URL/type, version pin, effective date, instrument family. Parser dispatches on source. |
| Language | English month names in `chunker._format_effective_date`; all prompts (`RAG_PROMPT`, `GOLDEN_PROMPT`, judge criteria); abstention string `"The corpus does not address this."` (matched by judge) | Language profile: PT prompts + PT months + PT abstention phrase (`"O corpus não aborda esta questão."`) — must change **atomically** across `prompts.py` + `judge.py` + `golden_generator.py`, else judge/eval silently mis-scores. These files carry `E501` ignores precisely because rewrapping moves eval numbers — same caution applies. |
| Embedder | `BAAI/bge-small-en-v1.5` in `embedder.py` (14 LOC); `VECTOR_SIZE = 384` in `vector_store.py` | Model name + vector dim from config (multilingual, e.g. bge-m3 class, 1024-d). Derive dim from model metadata, assert vs collection — a silent 384/1024 mismatch is the obvious fork bug. New `COLLECTION_NAME` per corpus profile. |
| Chunker grammar | `REGULATION_NAMES` (ai_act/nis2); header templates `"Article {n}, {regulation_name}, effective {date}"` in `_make_header` | Instrument-name table + PT header grammar: `"Artigo 12.º, Decreto-Lei n.º 127/2024, de 24 de agosto"`; families: Decreto-Lei, Lei, Portaria, Decreto Regulamentar, Regulamento. Diploma-citation variants (`n.º`, `.º`, Roman chapters) live here. |
| Eval tiers | `run_eval.py` `choices=["lepanto","demo"]`; `corpus_pins.check_record_pins(record, ai_act_path, nis2_path)` — **two-file signature baked in** | Tier list from config; `corpus_pins` generalized to `{name: path}` dict so `anchor_terms` maps term → per-diploma hit counts. Same for `verify_unanswerable_pins.py`. |
| LLM | `MODEL="MiniMax-M3"`, minimax base_url in `llm.py` | Env-driven model/base_url (keep MiniMax as default; `<think>` stripping is generically useful). |

### Rewrite (new-code work, not translation)
1. **`ingestion/dre.py` replaces `eur_lex.py`.** The EUR-Lex parser is 100% coupled to OJ/ELI CSS classes (`.eli-subdivision[id^='rct_']`, `[id^='art_']`, `div[id^='cpt_IVXLCDM']`, `.oj-ti-art`, `.oj-doc-ti`, `.eli-container[id^='anx_']`). DRE HTML shares none of this. Rewrite against DRE's actual markup; keep the unit schema (`regulation→diploma`, `part`, `article_num`, `annex_id`, `chapter`, `celex→dre_ref`, `effective_date`, `content_hash` inputs) so downstream (chunker, vector_store, golden machinery) survives.
2. **Annex/table chunking.** `chunker._split_annex_rows` splits on blank lines — fine for AI-Act annex prose, destructive for PT emission-limit tables (valores limite de emissão, Portaria-style annexes full of numeric tables: chemical-industry core content). Rewrite: linearize table rows with repeated column headers as row-prefix context, keep the 800-token packing loop (`_chunk_long_annex`) which is sound.
3. **`download_corpus.py` → manifest-driven fetcher** with per-diploma pinned URLs (original vs consolidated version — see risks), sha256 recording on download (already prints it; now persist it for pins).
4. **`eval.yml` corpus steps**: CELEX downloads + inline `[(celex, name, out)]` loop → manifest-driven; baseline URL → pt-amb-rag repo; keep the "points == chunks" assertion and keyless `--no-llm` gate (local embedder ⇒ CI needs no API key — preserve, it's the fork's cheapest quality win).

### Drop
- `data/raw/*.html`, `data/processed/*.jsonl` (gitignored, regenerated), all `evals/golden/*.jsonl` + `_vetting_state/` (tied to EN content hashes — regenerate), `evals/results/*` incl. baseline.
- `ingestion/eur_lex.py`, CELEX machinery after extraction (keep for reference in git history).
- `TUTORIAL.md` / `TUTORIAL/` EU-specific walkthroughs (rewrite in Phase 6 if the fork deserves one).
- AI-Act/NIS2 prompt framing ("EU regulation" persona) — replaced, not ported.
- `playground.py`: port cheaply (title/labels) or drop; decide at Phase 6 — it's a single self-contained file.

---

## 2. New modules

| Module | Purpose | Notes |
|---|---|---|
| `ingestion/dre.py` | Parse DRE HTML/XML into the unit schema | Spike first (Phase 1); prior art: `apiaberta/connector-dre` (community DRE connector) and dados.gov.pt open-data distributions — check before hand-rolling scraping. |
| `ingestion/citation_parser.py` | **PT legal-citation parser**: normalize `artigo 12.º do Decreto-Lei n.º 127/2024`, `n.º 2 do artigo 8.º`, `alínea b) do n.º 3`, `Portaria n.º 38/2025`, Roman chapters → canonical `(diploma, artigo, número, alínea)`. | Three uses: (a) payload `refs` field at chunk time (Phase 1, minimal); (b) golden curation — expand `all_acceptable_ids` with citation targets, directly attacking the original's known limitation "cross-references broken at chunk level" (BUILD_PLAN.md line 168); (c) judge `citation_accuracy` grounded in PT citation grammar instead of free-text. |
| `config.py` + `corpora.yaml` | Corpus manifest, language profile (prompts/months/abstention/citation templates), embedder profile (model, dim, collection) | Single source of truth replacing the four hardcode clusters above. |
| `eval/corpus_pins.py` (generalize) | N-diploma signature; `anchor_terms`/`must_be_zero` semantics unchanged | The unanswerable-tier design ports perfectly: PT unanswerables = e.g. "what does Spain's REACH authority require" / installation-specific SILIAMB process questions. |
| `ingestion/siliamb.py` (**deferred, verify first**) | SILIAMB/COMAPER metadata enrichment: installation CAE codes (20/21 = chemicals), license-process status, linked diploma references | Treat as *metadata enrichment*, never a corpus source (process data is per-customer, likely not openly reusable). Blocked on the legal-landscape agent confirming public availability and on INCM/APA terms. Do not let it near the Phase 1–4 critical path. |
| PT prompt set | `RAG_PROMPT` PT, citation format `[Decreto-Lei n.º X, Artigo Y.º]`, `GOLDEN_PROMPT` PT, judge criteria PT | Keep E501-ignore discipline + "don't rewrap prompts" rule — byte-stable prompts are what make eval numbers comparable. |

Corpus targets (candidate list — **pending confirmation by the legal-landscape agent**): DL 127/2024 (Licenciamento Único de Ambiente, replacing DL 127/2013 REI), the Portaria approving emission-limit-value regulations (large PTE tables), REGEP (waste), AIA regime, Seveso III transposition (DL 150/2015 — chemical-industry major-accident), water-resources regime. Engineering constraint: manifest-driven from day one, pilot with ONE diploma so the pipeline isn't blocked on legal scope.

---

## 3. Repo/config strategy — recommendation: **fork with cherry-picks**

Options weighed against this codebase's actual shape:
- Genuinely generic code (metrics, sampling, prefilter, gate, vector-store mechanics, run_eval runners) ≈ 800–1,000 LOC.
- Corpus/language-coupled code (eur_lex, chunker grammar, all prompts, corpus_pins signature, CI corpus steps) ≈ the rest of the load-bearing logic.

**Shared-core package** loses: at this size, packaging/versioning/publish + cross-repo sync PRs cost more than duplicating ~900 LOC; the divergent half (ingestion, prompts) can't be shared anyway without a speculative plugin ABI.
**Plugin packages** lose: same ABI speculation; only two consumers would exist.
**Fork + documented cherry-pick contract** wins: full copy, rename `safety_rag`→`pt_amb_rag`, parameterize the four axes, and keep upstream-portable files (`eval/metrics|sampling|prefilter.py`, `eval_gate.py`, `run_eval*.py` skeletons, `ci.yml`) **minimally diffed** so upstream fixes (e.g. the run_eval k_rank bugfix lineage already visible in its docstring) cherry-pick cleanly. Record the contract in a `UPSTREAM.md` listing portable paths.

**Decision trigger to revisit:** if a third jurisdiction fork appears, extract `legalrag-core` (metrics + gate + runners + vector-store) and make both forks depend on it. Not before.

---

## 4. Phase plan (eval loop BEFORE retrieval upgrades — same discipline as BUILD_PLAN.md line 39)

| Phase | Deliverable | Hours | Gate |
|---|---|---|---|
| **0** | Fork, rename to `pt_amb_rag`, pyproject/CI port, ruff+pyright+pytest green on empty corpus, `UPSTREAM.md`, `corpora.yaml` schema (manifest with 1 pilot diploma) | 3–4 | ci.yml green with ported tests |
| **1** | DRE ingestion: `dre.py` parser for pilot diploma, PT chunker grammar + table-aware annex splitting, minimal citation parser → payload `refs`, manifest-driven download with recorded sha256s, parser fixture tests (committed HTML snapshots) | 8–12 | units parsed end-to-end; pins reproducible across two fresh downloads |
| **2** | Retrieval v1: multilingual embedder from config (dim asserted), `build_index.py` port, `ask()` smoke answers in PT | 3–4 | points == chunks assertion locally |
| **3** | API + Docker port (title, collection, env): mostly config — auth, routes, threadpool, `--no-llm` path unchanged | 2–3 | `/health`, `/search`, `/ask` live in compose |
| **4** | **Eval loop (the centrepiece, before any retrieval upgrade):** PT golden-gen + judge prompts, `corpus_pins` N-diploma generalization, tier config, generate + hand-curate golden sets (~30–40 questions main tier, ~12 unanswerable with anchor terms), commit baseline, `eval.yml` gate ±0.02 vs baseline with fresh DRE download in CI | 10–14 (≈3–5 of them human curation) | baseline committed; gate blocks PRs; unanswerable pins verified |
| **5** | **Conditional on Phase 4 < pre-registered threshold:** hybrid BM25(PT stemming/stopwords)+dense+RRF and/or reranker. Original baseline was Hit@5 0.806 vs pre-registered 0.85 → expect PT to need this too (multilingual embeddings, noisier corpus) | 5–7 | gate must improve or Phase 5 is reverted |
| **6** | Citation parser full integration (judge citation_accuracy, `all_acceptable_ids` expansion), README/BUILD_PLAN rewrite (PT fork story), playground/scope decisions, launch post | 4–6 | generation eval (faithfulness, abstention) green |

**Total ≈ 35–50 h.** Pre-register the Hit@5 threshold **before** first Phase-4 run (the original's defensible-numbers rule, BUILD_PLAN line 89). LLM cost stays in the ~€5/cycle regime (local embeddings in CI unchanged; golden-gen + judge only in Phase 4/6).

---

## 5. Risks + mitigations

1. **DRE corpus availability/quality** — no ELI/OJ-style guaranteed XHTML; markup varies by era; some annexes PDF-only. *Mitigation:* Phase-1 spike timeboxed; check API Aberta connector + dados.gov.pt distributions before scraping; committed HTML fixtures per diploma; PDF-only annexes recorded as limitations (BUILD_PLAN precedent), never silently skipped.
2. **Consolidated-text drift** — DRE consolidated versions update without notice; original vs consolidated changes answers. *Mitigation:* manifest pins exact version/URL per diploma + snapshot date; CI re-downloads fresh each run (eval.yml pattern) and `corpus_pins` sha256s fail loudly on drift — this machinery already exists, port it per-diploma.
3. **DRE licensing** — reuse of official-journal content is broadly enabled by the PT open-data regime (Lei 26/2016 lineage, EU Directive 2019/1024 transposition visible on diariodarepublica.pt itself), but INCM-produced consolidated editions may carry their own terms, and bulk scraping the live site likely violates usage terms. *Mitigation:* prefer the official open-data distribution; verify INCM reuse terms **before** any public/customer-facing corpus-derived index ships; keep raw HTML internal like the original does (data/ gitignored, rebuild-from-source).
4. **Bilingual eval cost & judge quality** — all PT prompts via MiniMax: cost unchanged, but LLM-as-judge in PT legal language is unproven. *Mitigation:* ~20% human-judge baseline (already the original's policy, BUILD_PLAN line 170); hand-label a 15–20 question calibration set before trusting the judge; tier sizes capped (31–40 q).
5. **Embedder swap regression** — EN→multilingual changes dims + ranking behavior; silent collection mismatch possible. *Mitigation:* dim derived from model and asserted at collection creation; new collection name; full re-baseline (never compare across embedders).
6. **Ranking, not recall, is the expected failure mode** — original: Recall@10 0.968 but Hit@5 0.806 (recital near-duplicates, cross-refs). PT port should expect the same shape → that's why Phase 5 is hybrid/rerank, not more chunking.
7. **Dependency on parallel legal agent** — corpus scope isn't mine to finalize. *Mitigation:* manifest-driven design + one pilot diploma keeps phases 0–4 executable now; widening the corpus is a manifest edit + golden-set extension, not a code change.

---

### Execution notes
- Ordering invariant: no retrieval-quality changes (Phase 5+) until the Phase-4 baseline exists and is committed — this is the fork's inherited core discipline.
- Prompt files stay byte-stable after authoring (E501 ignores preserved): rewrapping PT prompts would silently move judge/golden numbers.
- `corpus_pins` two-file signature and `run_eval` hardcoded tier list are the only two upstream files needing structural edits — flag both in `UPSTREAM.md` as permanent cherry-pick conflict zones.
