# RAG Methodology Upgrade Plan — PT-Legal Fork of `eu-ai-rag`

Target: fork processing **Portuguese-language legislation** (Decreto-Lei/Lei/Portaria from the Diário da República; EU directives may have EN summaries but the national corpus is PT).
Current baseline: article-aware chunking + contextual headers, `bge-small-en-v1.5` dense-only, Qdrant, MiniMax-M2 generation, golden-set Hit@5 = 0.806 / MRR@10 = 0.761, CI gate ±0.02.

Two hard constraints discovered in the repo that shape this plan:

1. **The embedding model is English-only.** `bge-small-en-v1.5` never saw Portuguese in training; every PT query and every PT chunk is out-of-distribution. This is the single largest defect and the first thing to fix.
2. **The gate is statistically underpowered.** The active golden set is `demo.jsonl` (36) + `lepanto.jsonl` (31) + `unanswerable.jsonl` (12) ≈ 79 items. At n≈60–79, the 95% CI on a Hit@5 near 0.85 is **±0.08–0.09** (binomial SE = √(0.85·0.15/n)) — the ±0.02 tolerance in `scripts/eval_gate.py` is far inside the noise band. You need n ≈ **1,225** for ±0.02 at 95% confidence, or paired significance tests on per-question results. Eval upgrades are not optional polish; they are prerequisite to claiming any of the deltas below.

---

## 1. Multilingual / PT embeddings

### Benchmark evidence (Portuguese rows, primary sources)

**MIRACL dev, PT, nDCG@10** — from the BGE-M3 paper Table 2 [2] (PT is a MIRACL language: 6,569 corpus docs, 200 dev / 200 test queries [20][2]):

| Retriever | PT nDCG@10 |
|---|---|
| BM25 (Lucene) | 44.9 |
| mDPR | 65.5 |
| mContriever | 72.0 |
| mE5-large | 73.5 |
| OpenAI-ada-002-era (OpenAI-3) | 73.7 |
| **bge-m3 dense** | **76.3** |
| bge-m3 multi-vec (ColBERT) | 76.4 |
| bge-m3 dense+sparse | 76.5 |
| E5-mistral-7b | 77.5 |

**MKQA cross-lingual, PT target, Recall@100** — BGE-M3 paper Table 3 [2]: BM25 31.8 · mE5-large 66.8 · **bge-m3 dense 69.3** · dense+sparse 69.8 · all 70.6. Cross-lingual (EN query → PT corpus) works with bge-m3 at only ~7 points below monolingual dense.

**mE5 family average** (tech report [4]): MIRACL avg nDCG@10 — small 57.9, base 59.5, large 61.5. mE5-small/base are solid but a class below bge-m3 on PT.

**PT-legal specifics (JurisTCU, 16,045 BR-PT legal docs, 150 judged queries [13])**: BM25 P@10 = 23.8; OpenAI embeddings 37.8–40.8 (~+70%); docT5query expansion 33.6 (+41%); notably **BERT-pt-large-legal (a PT-legal-adapted encoder) scored 20.2 — *below* plain BM25**. Lessons: (a) dense retrieval is strongly superior to BM25 on PT legal text when the encoder is strong; (b) small legacy "PT-legal BERT" adaptations do **not** beat modern multilingual retrievers — do not chase LegalBert-pt [22] for the retrieval path; (c) lexical + expansion methods remain a useful complement.

### Candidates

| Model | Params / dims | Ctx | License | PT evidence | Verdict |
|---|---|---|---|---|---|
| `BAAI/bge-m3` [1] | ~568M (XLM-R-large) / 1024 | 8192 | MIT | MIRACL PT 76.3 dense; MKQA PT 69.3; dense+sparse+colbert in one model | **Recommended** |
| `intfloat/multilingual-e5-base` [3][4] | ~278M / 768 (12 layers) | 512 | MIT | MIRACL avg 59.5 (PT ~5–7 pts below bge-m3) | CPU-constrained fallback |
| `intfloat/multilingual-e5-small` [3] | ~118M / 384 (12 layers) | 512 | MIT | MIRACL avg 57.9 | Only if bge-small latency must be preserved exactly |
| `jinaai/jina-embeddings-v3` [5] | 570M / 1024 (Matryoshka 32–1024) | 8192 | **CC-BY-NC-4.0** | 89 langs, task LoRA | **Blocked for commercial use** unless licensed |
| `Qwen/Qwen3-Embedding-0.6B` [6] | 0.6B / 1024 | 32k | Apache-2.0 | MTEB-multilingual avg 64.33 (vs Cohere-embed-v3 61.12, Gemini 68.37) | Strong dark-horse; heavier inference (decoder arch), newer, less PT-legal track record |
| `BAAI/bge-reranker-v2-m3` [7] | ~568M cross-encoder | — | Apache-2.0 | multilingual (MIRACL eval in card) | Reranker, not retriever — §2 |

### Recommendation: `BAAI/bge-m3`

Reasoning:
1. **Best per-language PT evidence of any open model** — 76.3 nDCG@10 on MIRACL PT dense, and the only model with published PT numbers for dense, sparse, and their combination [2].
2. **MIT license** vs jina-v3's CC-BY-NC (commercial blocker) [1][5].
3. **8192-token context** means a whole PT artigo with its numbers/alíneas embeds in one vector — no truncation mid-provision, which is exactly where citation-critical text lives. mE5 caps at 512 tokens and silently truncates long artigos.
4. **One model, three retrieval modes** (dense + learned sparse + ColBERT) [1] — you can light up Qdrant hybrid without adding a second indexing stack, and the repo README already anticipated bge-m3 as the swap target.
5. **E5-style prefixes not needed** (bge-m3 needs no `query:`/`passage:` prefixes — one less footgun than mE5, which requires them even for non-English [3]).

Tradeoff: ~17× the parameters of bge-small-en (33M → ~568M). On CPU, expect query embedding latency to go from ~5–10 ms to ~100–300 ms (estimate from parameter count, not measured). Mitigations: ONNX/quantized int8 export (≈2–4× CPU speedup), batch index-time embedding, or serve on a modest GPU. If CPU-only hard constraint: `multilingual-e5-base` (278M) keeps most of the PT gain at ~half the cost, accepting 512-token truncation on long artigos (cap chunk size accordingly).

**Expected delta**: switching EN-only → bge-m3 on a PT corpus is the largest single win. Bracket: MIRACL PT BM25=44.9 vs bge-m3=76.3 shows the EN/lexical gap on PT; an EN-only encoder on PT behaves far closer to the bottom of that bracket. Estimate Hit@5 0.806 → **0.85–0.90** (to be confirmed on your golden set — treat as hypothesis, gate it).

### Cross-lingual retrieval (EN query over PT corpus)

Three options, in order of cost:
1. **Zero-shot cross-lingual with bge-m3** — MKQA PT shows dense 69.3 Recall@100 with EN queries [2]. Free, already in the stack. Start here.
2. **Query translation via MiniMax-M2** before retrieval (EN→PT), one cheap LLM call (~€0.30/M input [repo README]); multilingual generative query expansion reports gains up to ~10–15% in cross-lingual settings [18].
3. **HyDE in PT** — generate a hypothetical PT article with the LLM, embed it, retrieve doc-doc similarity; strong multilingual results in the HyDE paper [17], and LLM generation cost is trivial at your scale.

Add a dual-language header on each chunk (PT title + EN gloss of the diploma) so EN queries lexically match translated instrument names ("Decree-Law 127/2013" ↔ "Decreto-Lei n.º 127/2013").

---

## 2. Hybrid retrieval: BM25(PT) + dense + RRF

### Why: evidence
- Anthropic: contextual embeddings + contextual BM25 cut top-20 retrieval failures **49%**; adding a reranker, **67%** [11].
- 2026 mixed text+table benchmark (23k queries): two-stage hybrid + neural rerank was best of 10 strategies, Recall@5 = 0.816, "outperforming all single-stage methods by a large margin" [16].
- On MIRACL PT the dense→hybrid delta is small (76.3→76.5) [2] — **Wikipedia-like text is dense-friendly; legal text is more lexically anchored** (exact artigo numbers, defined terms "entidade gestora", ELV values), so expect the hybrid delta on *your* corpus to be above the MIRACL delta. JurisTCU shows BM25 remains a strong complement on PT legal (MRR@10 53.9 baseline) [13].

### Implementation (concrete)

**BM25 for PT** — two supported routes:
- **Qdrant-native**: server-side BM25 sparse vectors + `prefetch` + `models.RrfQuery(rrf=models.Rrf())` — the exact Query API shape is documented [8] and matches your `qdrant-client` usage in `vector_store.py`. One service, one query, RRF and DBSF built in.
- **`bm25s`** (client-side, Numpy, numba ~2× speedup, no Java) [9] if you want corpus-side control; supports custom stopword lists and Snowball stemming via PyStemmer [9]. Snowball has a `portuguese` stemmer (standard `nltk`/PyStemmer package).

**PT tokenization pipeline for BM25** (critical — default English analyzers will mangle PT):
- Stopwords: PT list (spaCy `pt_core_news_lg` ships Portuguese lang data [10], or NLTK's `portuguese` list; both public).
- Stemming: Snowball Portuguese (`PyStemmer`, used by bm25s [9]) for recall on inflection ("registo/registos/registada"). Light alternative: spaCy lemmatizer (`pt_core_news_lg`, trainable lemmatizer over UD Portuguese Bosque [10]) — better precision, slower (full pipeline). Recommendation: Snowball for BM25 index, spaCy lemmas later if you add a lexical-match feature.
- **Accent/diacritic folding**: index both `ação` and `acao` (PT queries frequently omit diacritics). Qdrant/bm25s won't do this by default — add a normalizer in the tokenizer.
- **Legal-term protection list**: do NOT stem `artigo`, `n.º`, `alínea`, `anexo`, DL numbers — exact-match them. Numbers ("127/2013") must be indexed verbatim.

**Fusion**: RRF (rank-based, no score calibration needed) via Qdrant Query API `prefetch` dense limit=20 + sparse limit=20 → `RrfQuery` limit=10 [8]. If you later want weighted fusion, Qdrant's `Formula` query or DBSF [8].

**Expected delta**: +0.02–0.04 Hit@5 over dense-only on legal QA (defensible given [11][16] and the lexical anchors in legal text; MIRACL PT's +0.2 [2] is the pessimistic Wikipedia-like bound).

### Reranker: `bge-reranker-v2-m3` — when it pays off

- **When**: after hybrid recall is high (Hit@20 ≥ 0.95) but ordering is imperfect (MRR stuck), or when the generator is sensitive to context order/noise (it is — your prompt quotes sources inline). The 2026 benchmark shows reranking is what pushes a hybrid system from good to best-in-class [16].
- **Cost**: ~568M cross-encoder, one forward pass per (query, chunk) pair → reranking 20 pairs ≈ 0.5–2 s CPU / ~50–100 ms GPU (estimate). Run it only on the top-20 fused candidates, only in the API path — **skip it in CI retrieval evals** to keep the loop fast, but include a nightly reranker-on eval.
- **Caveat (evidence!)**: on LegalBench-RAG, a *general-purpose* reranker (Cohere English) **hurt** precision on legal text vs no reranker [12]. bge-reranker-v2-m3 is multilingual and stronger, but the lesson stands: **gate the reranker on your PT golden set before shipping**; expected gain +0.02–0.05 MRR@10 if it works, negative if domain mismatch bites.
- Alternative with a PT track: `Qwen3-Reranker-0.6B` (Apache-2.0, same family as Qwen3-Embedding [6]) as a second candidate to A/B.

---

## 3. PT legal-domain chunking

The repo's chunker (`src/safety_rag/ingestion/chunker.py`) is article-aware with headers and content-hash versioning — good bones. Generalize it from EU-regulation shape to PT diploma shape.

### Structural hierarchy to parse (regex-first, AKN-PT where available)

| Unit | PT patterns | Chunk policy |
|---|---|---|
| Diploma header | `Decreto-Lei n.º 127/2013`, `Lei n.º 58/2013`, `Portaria n.º …`, "de 30 de agosto" | metadata, always in contextual header |
| Título / Capítulo / Secção | `TÍTULO I`, `CAPÍTULO 2.º`, `Secção II` | breadcrumb in header |
| **Artigo** | `Artigo 8.º`, `Artigo único` (with following epigraph line) | primary chunk unit (keep whole; bge-m3's 8k ctx makes splitting rare) |
| **Número** | `1 - ...` / `1. ...` / `n.º 2` | sub-chunk only if artigo > ~700 tokens; record parent |
| **Alínea** | `a) ...` through `z)`, `aa)` | never split alone; keep with número; store `alinea` field for citation |
| Parágrafo/Inciso | `— ...` indented lines | attach to alínea |
| **Anexo** | `ANEXO I`, `Anexo A`, "a que o artigo 4.º refere" | special table handling (below) |
| Considerandos | `Considerando que:` preâmbulo | one chunk for preamble; low retrieval weight |

Store machine fields: `{diploma_type, diploma_number, diploma_year, date_published, artigo, numero, alinea, anexo, epigraph}` — these feed both Qdrant payload filters and citation-aware generation.

### Anexos with tables of limit values (the emission-ELV case)

This is where PT environmental law lives (e.g., ELV tables in anexos of environmental DLs) and where naive chunking destroys retrievability:
1. Detect table blocks in the DRE HTML (DRE serves HTML for every diploma via ELI URIs, PDF also available [15]) — do **not** flatten tables to prose before chunking.
2. Serialize each table to **Markdown table syntax** (pipes preserved) — embedding models and the generator both read MD tables well, and BM25 keeps row terms.
3. **Chunk by row-group, not by table**: one chunk = table caption + column headers + up to ~15 rows (~500 tokens). Every row-chunk repeats the full column-header block (units! `mg/Nm³`, `µg/l`) and the anexo/article pointer. A query "valor limite de emissão de SO2 numa instalação de combustão" must match *one row group*, not a 3,000-token table blob.
4. Number-rows ("01 02 03" CNE codes) and substance names ("dióxido de enxofre (SO2)") go in the chunk **and** as payload keywords for exact BM25 matching.
5. Add a synthetic "table summary" line at the top of each table chunk (e.g. "Anexo III do DL n.º 127/2013: valores limite de emissão de poluentes para instalações de combustão, por potência e combustível") — a poor-man's Anthropic-style contextual header [11] generated deterministically from metadata, no LLM needed.

### Contextual header strategy for PT legal citations

Extend `_make_header` to emit PT-native breadcrumbs, both languages of the instrument name:

```
Decreto-Lei n.º 127/2013, de 30 de agosto (DL 127/2013) — alterado pelo DL n.º 68/2015
> TÍTULO II, CAPÍTULO 3, Artigo 8.º (Licenciamento), n.º 2, alínea b)
> Anexo: —  [consolidação de 2024-01-01]
> Fonte: Diário da República, 1.ª série, ELI data.dre.pt/eli/dec-lei/127/2013/p/cons/20240101/pt
```

Why: (a) the header text is indexed by both dense and BM25, so every chunk carries its citation identity — "qual o n.º 2 do artigo 8.º do DL 127/2013?" becomes a lexical+dense easy match; (b) the amendment chain and consolidation date in the header make the generator's citation job trivial and auditable; (c) per Anthropic, contextual headers cut retrieval failures ~49% combined with BM25 [11].

### Amendments and consolidated text (versioning)

- DRE exposes **consolidated texts at versioned ELI URIs**: `data.dre.pt/eli/{type}/{number}/{year}/{region}/cons/{yyyymmdd}/pt/html` [15]. Parse the `cons` date into `consolidated_at`; treat it as the corpus version key.
- **CRITICAL legal caveat**: DRE consolidated texts are marked `legal_value: unofficial` ("texto consolidado sem valor legal") [15]. Every consolidated chunk must carry `is_consolidated: true`, and generation must disclose it (§4).
- Amendment graph: parse "alterado pelo / revogado pelo" lines in the diploma text; even better, `dre.tretas.org` publishes weekly SQLite dumps with a `dreapp_document_connects_to` table that already encodes document-to-document links (amendments/revocations) [21] — use it to populate `amended_by` / `revoked_by` payload fields for filtering and freshness checks.
- Keep the **original published text AND the consolidated text** as separate chunks (different `text_version` payload), original for citation-fidelity, consolidated for "what's the law today" queries. Filter default = latest consolidation.
- Content-hash versioning (already in the chunker) extends naturally: `sha256(header + text + consolidated_at)`.

---

## 4. Legal-citation-aware generation

### Prompt structure (PT-native citation grammar)

Replace `RAG_PROMPT` in `generation/prompts.py` with a PT-legal variant. The current `[Regulation X, Article Y]` grammar doesn't map to PT citation practice. Force the full citation grammar:

```
Cite assim: [DL n.º 127/2013, art. 8.º, n.º 2, al. b)] ou [Lei n.º 58/2013, art. 3.º] ou
[Portaria n.º X/2017, anexo I, tabela 2]. Nunca cites só o diploma sem o artigo.
Se a fonte é uma tabela de valores: cita [DL ..., anexo III, tabela 1] e reproduz a linha
relevante com unidades.
```

Full prompt skeleton (PT, since corpus and users are PT):

- **System**: "Assistente de compliance regulatório sobre legislação portuguesa… Responde em português europeu (PT-PT), exceto se a pergunta for noutra língua — nesse caso responde nessa língua mas mantém as citações em português."
- **Citation rule** (above).
- **Amendment rule**: "Se o texto do chunk indica que o artigo foi alterado ('alterado pelo DL X'), cita ambas: a redação atual [DL base, art. N] e o instrumento de alteração [DL X, art. N], e escreve '(redação introduzida pelo DL n.º X)'."
- **Consolidated-text disclaimer** (mandatory footer when any cited chunk has `is_consolidated=true`): "⚠ Resposta baseada em texto consolidado do DRE com data de consolidação {date}. O texto consolidado não tem valor legal; só o Diário da República original o tem. Verifica em dre.pt antes de decisões vinculativas." — grounded in the ELI register's `legal_value: unofficial` [15].
- **Abstention** (PT): reply exactly "O corpus não responde a esta pergunta." — keep the deterministic sentinel so the unanswerable tier stays machine-checkable.
- **Table rule**: when a source chunk is a table row-group, reproduce the row(s) verbatim with units; never recompute or merge values.
- **Context format**: keep `[Source N | DL n.º … | art. … | n.º … | al. … | anexo … | consol. YYYY-MM-DD | oficial/consolidado]` headers — extend `_format_source_header` with the new PT fields (`diploma`, `artigo`, `numero`, `alinea`, `anexo`, `consolidated_at`, `text_version`).

### Amendment handling beyond the prompt
- Retrieval should prefer `text_version=consolidated` by default but return the original alongside when `amended_by` is non-empty — the prompt then has both texts and can produce the dual citation without hallucinating the amendment chain.
- Never let the generator *infer* amendments from its own knowledge — only from chunk headers. Add a judge criterion for it (§5).

---

## 5. Eval upgrades

### Golden-set design for PT legal QA — four tiers

1. **PT→PT core tier (n ≥ 150)**: PT questions over PT corpus. Stratify by target unit: artigo-level (60%), n.º/alínea-level (20% — the hard cases that prove hierarchy parsing), anexo/table-level (20% — ELV lookups; gold = specific row + value + unit). Use the three JurisTCU query styles as a template — real-user keyword queries, keyword queries, and full-sentence questions behave very differently (BM25 vs dense rankings diverged by query type [13]).
2. **Cross-lingual tier (n ≥ 50)**: EN questions over the PT corpus (EN-question/PT-corpus), e.g. "What are the emission limit values for SO2 in the annex to Decree-Law 127/2013?" — measures the bge-m3 MKQA-style path [2]. Report separately; do not average into the PT score.
3. **Unanswerable-PT tier (n ≥ 40)**: three sub-designs:
   - *Near-miss instruments*: "art. 12.º do DL 127/2013" when art. 12 doesn't exist / has no n.º 4;
   - *Neighbouring-law traps*: concepts that exist in the EU directive's EN summary but have no PT transposition in corpus;
   - *Post-cutoff / future amendments*: questions about amendments after your `consolidated_at` date — tests the freshness disclaimer behavior.
   Keep the deterministic sentinel check; add a citation-format regex gate (below).
4. **Multi-article tier (n ≥ 30)**: questions spanning 2+ artigos or 2 diplomas ("compara os requisitos do art. 8.º do DL A com o art. 4.º da Portaria B") — exercises query decomposition (§6) and multi-source citation.

### Metrics & statistics (fix the gate)
- Keep Hit@5 / MRR@10 / Recall@10 per tier; **never pool tiers**.
- Add **citation-accuracy as a deterministic metric**: regex-extract `[DL n.º …, art. …, n.º …, al. …]` from the answer and check each against the retrieved chunk payloads — no LLM needed, CI-safe. Target ≥ 0.95.
- Replace the ±0.02 absolute gate with: (a) report Wilson 95% CI per tier at current n; (b) gate on **paired bootstrap or McNemar/permutation test** over per-question outcomes (current vs baseline, same questions) at p < 0.05 — this is valid at n = 150, whereas the ±0.02 band needs n ≈ 1,225 (binomial math, computed).
- Keep the LLM judge for faithfulness, but see below.

### LLM-judge considerations for PT output
- Empirical warning: multilingual LLM-as-judge consistency across languages averages **Fleiss' κ ≈ 0.3** across 5 models / 25 languages — judges are materially less consistent in non-English [14]. Mitigations:
  - **Judge in the same language as the answer** (PT) with an explicit PT rubric, not an EN rubric applied to PT text;
  - Keep verdicts coarse (yes/no/partial — already your shape in `judge.py`) — fine-gramed scores degrade faster across languages;
  - Add a one-time **judge-consistency audit**: run 30 questions in PT and their EN translations, measure agreement; if κ < 0.5, prefer the deterministic citation check + human spot-check over the judge for release decisions;
  - Pin judge prompts + temperature (already 0.2) and store judge outputs per run for audit.

---

## 6. Further upgrades that matter for legal RAG

### Query decomposition for multi-article questions
LLM decomposes the question into sub-questions (one per artigo/diploma), retrieve per sub-query, merge + dedupe, rerank the union — exactly the decomposition→merge→rerank pipeline shown to help multi-hop legal-style questions [19]. Implement as a single MiniMax-M2 call returning JSON `[{sub_q, hint_filter}]`; only trigger when the question contains ≥2 distinct citation patterns (regex on `\b(artigo|DL|lei|portaria|anexo)\b` count ≥ 2) to avoid paying latency on simple queries. Eval on the multi-article tier (§5.4).

### Metadata filtering by instrument/year
You already filter by `regulation` in the API. Extend Qdrant payload filters to `diploma_type`, `diploma_year`, `artigo`, `anexo`, `text_version`, `in_force`. Two payoffs: (a) precision — "no DL 127/2013, o que diz o artigo 8.º?" becomes a filter + tiny rerank, near-100% Hit@5; (b) eval isolation — run per-instrument slices to localize regressions. Extract the filter intent with a small deterministic regex first (citation patterns are regular), LLM only as fallback.

### Freshness & versioning of consolidated legislation
- Weekly job: pull the tretas.org SQLite dump (Sunday export [21]) or diff DRE ELI pages for your diploma list; detect `consolidated_at` changes; re-chunk + re-embed only changed diplomas (content-hash already gives you the change detector);
- Store per-diploma `latest_consolidated_at` and expose it in `/ask` responses ("fonte: consolidação de 2026-09-28");
- Unanswerable sub-tier for post-cutoff questions doubles as the freshness regression test.

### Contextual embeddings (deterministic, not LLM)
Your headers already approximate this; for PT legal the deterministic breadcrumb header (§3) is nearly free and — combined with BM25 — is where Anthropic's 49% failure reduction came from [11]. Skip LLM-generated chunk contexts until the deterministic version is measurably exhausted.

### Do NOT adopt (evidence-based)
- **LegalBert-pt / BR-PT legal encoders** as retrievers — underperformed even BM25 on PT legal IR [13][22]; BR-PT vs PT-PT mismatch compounds it.
- **jina-embeddings-v3** — CC-BY-NC license blocks commercial use [5].
- **Unconditional reranking** — general rerankers can hurt on legal text [12]; gate on golden set.

---

## 7. Phased rollout (each phase gated on the upgraded eval)

| Phase | Change | Files | Expected Δ Hit@5 (PT→PT tier) | Risk |
|---|---|---|---|---|
| 0 | **Eval upgrade first**: 4-tier golden set (n≥270), paired-bootstrap gate, deterministic citation check, PT judge rubric | `evals/golden/*`, `scripts/run_eval.py`, `eval_gate.py`, `eval/judge.py`, `eval/metrics.py` | — (unblocks everything) | low |
| 1 | Re-baseline current system on PT tiers (expect a *drop* from 0.806 — that number was on EN corpus) | eval only | honest baseline | none |
| 2 | **bge-m3 dense** swap + PT breadcrumb headers + new chunker (artigo/n.º/alínea/anexo + table row-groups) | `retrieval/embedder.py`, `ingestion/chunker.py`, `scripts/build_index.py`, new `ingestion/dre.py` | +0.05–0.10 over Phase-1 PT baseline | re-embed cost, one-time |
| 3 | **Hybrid**: Qdrant BM25 sparse (PT stopwords + Snowball PT + diacritic folding + legal-term protection) + RRF | `vector_store.py`, `retrieval/bm25.py` | +0.02–0.04 | low |
| 4 | **Citation-aware generation** PT prompt + amendment/consolidation handling | `generation/prompts.py` | citation-accuracy 0.95+ (new metric) | prompt regression — judge-gated |
| 5 | **Reranker** bge-reranker-v2-m3 top-20, A/B vs Qwen3-Reranker-0.6B; keep only if paired test significant | `retrieval/rerank.py` | +0.02–0.05 MRR@10 (or revert) | may hurt [12] |
| 6 | Query decomposition + metadata filter intent + freshness job | `retrieval/decompose.py`, `api/ask.py` | multi-article tier Hit@5 +0.10+ (estimate) | latency; gate behind regex trigger |

Sequencing principle: **evals (0–1) before embeddings (2) before hybrid (3) before reranker (5)** — each phase needs the previous phase's measurement to be trustworthy; the current n=79 golden set cannot distinguish 0.806 from 0.87 (CI ±0.09).

---

## Sources

[1] https://huggingface.co/BAAI/bge-m3 — BAAI/bge-m3 model card
[2] https://arxiv.org/abs/2402.03216 — BGE M3-Embedding paper (MIRACL/MKQA/MLDR tables)
[3] https://huggingface.co/intfloat/multilingual-e5-small — intfloat/multilingual-e5-small model card
[4] https://arxiv.org/abs/2402.05672 — Multilingual E5 Text Embeddings tech report
[5] https://huggingface.co/jinaai/jina-embeddings-v3 — jina-embeddings-v3 model card
[6] https://huggingface.co/Qwen/Qwen3-Embedding-0.6B — Qwen3-Embedding-0.6B model card
[7] https://huggingface.co/BAAI/bge-reranker-v2-m3 — bge-reranker-v2-m3 model card
[8] https://qdrant.tech/documentation/search/hybrid-queries — Qdrant hybrid queries + RRF docs
[9] https://github.com/xhluca/bm25s — bm25s README
[10] https://huggingface.co/spacy/pt_core_news_lg — spaCy pt_core_news_lg model card
[11] https://www.anthropic.com/engineering/contextual-retrieval — Anthropic Contextual Retrieval
[12] https://arxiv.org/abs/2408.10343 — LegalBench-RAG paper
[13] https://arxiv.org/abs/2503.08379 — JurisTCU PT legal IR dataset paper
[14] https://aclanthology.org/2025.findings-emnlp.587 — How Reliable is Multilingual LLM-as-a-Judge?
[15] https://eur-lex.europa.eu/eli-register/portugal.html — EUR-Lex ELI register: Portugal (data.dre.pt)
[16] https://arxiv.org/abs/2604.01733 — From BM25 to Corrective RAG benchmark (2026)
[17] https://aclanthology.org/2023.acl-long.99 — HyDE paper
[18] https://arxiv.org/html/2511.19325 — Generative query expansion multilingual CLIR
[19] https://arxiv.org/abs/2507.00355 — Question Decomposition for RAG
[20] https://aclanthology.org/2023.tacl-1.63 — MIRACL dataset paper (TACL)
[21] https://blog.tretas.org/2024/06/16/obter-a-legislacao-do-dre-tretas-org-formato-sqlite — tretas.org weekly DRE SQLite dumps
[22] https://dl.acm.org/doi/10.1007/978-3-031-45392-2_18 — LegalBert-pt (BR-PT encoder)
