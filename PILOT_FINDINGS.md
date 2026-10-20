# Pilot findings — the DRE corpus spike

> What the Phase-1 corpus spike actually found, verified against the
> production Diário da República on 2026-10-20. This document supersedes
> the assumptions in [`Fork Engineering Plan.md`](./Fork%20Engineering%20Plan.md)
> §1 (Rewrite #1) and §5 (Risks 1–3).

Everything below was verified by running code against the live DRE, not
by reading documentation. The DRE publishes no documentation for the
interface used here.

---

## 1. The headline: DRE has no scrapable HTML — but it has a clean API

`diariodarepublica.pt`, `data.dre.pt` and `files.diariodarepublica.pt`
all serve the **same 2,346-byte OutSystems single-page-app shell** for
every path, including paths that should return a document, a PDF, or an
ELI HTML file:

```html
<!DOCTYPE html>
<html><head>…<script src="/dr/scripts/dr.appDefinition.js?…"></script>…</head>
<body><div id="reactContainer"></div><noscript>…</noscript></body></html>
```

The plan's Phase 1 assumed *"DRE HTML shares none of [the EUR-Lex CSS
classes]. Rewrite against DRE's actual markup."* — the premise was
wrong in a way that changes the work: **there is no markup to parse at
the URL level.** The document content is loaded client-side.

It is, however, available from the OutSystems **screenservice** that the
SPA itself calls. That API returns the fully structured document. The
contract, verified end-to-end:

| Step | Request | Response |
|---|---|---|
| 1 | `GET /dr/moduleservices/moduleversioninfo` | `{"versionToken": "…"}` — the module version |
| 2 | `GET /dr/scripts/dr.Legislacao_Conteudos.Conteudo_Detalhe.mvc.js` | the deployed bundle; contains the per-action `apiVersion` token and the screenservice URL |
| 3 | `POST /dr/screenservices/dr/Legislacao_Conteudos/Conteudo_Detalhe/DataActionGetAllConteudoDetalheData` | `301` + `Set-Cookie: nr2Users=crf%3D…` |
| 4 | same `POST`, replayed with `x-csrftoken: <crf>` | `200` + the document |

Step 4's payload is not HTML soup — it is the document, structured:

```
data.DetalheConteudo.Texto            # 116 KB of plain text (DL 150/2015)
data.DetalheConteudo.TextoFormatado   # 157 KB of the same document as HTML
data.DetalheConteudo.ELI              # https://data.dre.pt/eli/dec-lei/150/2015/08/05/p/dre/pt/html
data.DetalheConteudo.URL_PDF          # https://files.diariodarepublica.pt/1s/2015/08/15100/0537805401.pdf
data.DetalheConteudo.TipoDiploma      # "Decreto-Lei"
data.DetalheConteudo.Numero           # "150/2015"
data.DetalheConteudo.DataPublicacao   # "2015-08-05"
```

`TextoFormatado` **is** the "DRE markup" the plan was looking for, and
it is regular: only `<p>` elements and `<a>`/`<br>`, with five classes
— `paragraph-title-bold-center-18px` (the diploma title),
`paragraph-center` (structural heading), `paragraph-bold-center-14px`
(the heading's rubric), `paragraph-normal-text` (body), and
`paragraph-bold-center`. Article boundaries are therefore explicit:

```html
<p class="paragraph-center">Artigo 5.º</p>
<p class="paragraph-bold-center-14px">Deveres gerais do operador</p>
<p class="paragraph-normal-text">1 - O operador é responsável por …</p>
```

This is implemented in `src/pt_amb_rag/ingestion/dre.py`.

### Three transport details that are load-bearing

Each of these cost real debugging time and will bite again if changed:

1. **The screenservice URL separates module and screen with a slash**
   (`/screenservices/dr/Legislacao_Conteudos/Conteudo_Detalhe/…`), not
   the dot used in the screen's *script filename*
   (`dr.Legislacao_Conteudos.Conteudo_Detalhe.mvc.js`). With the dot,
   the request is not routed to the screenservice at all: the edge
   answers `301` with no `Set-Cookie`, and the session handshake can
   never complete. **This was the cause of a long false trail** — the
   symptom (a 301 loop) looks exactly like a session/CSRF problem.
2. **Redirects must not be followed.** The `301` points at `/dr/error`.
   Following it silently returns the 2,346-byte SPA shell, which parses
   as "no data" rather than as an error.
3. **CloudFront caches the cookie-less `301` for two minutes** and
   serves the cached copy — which has no `Set-Cookie` — to later
   clients. Each screenservice POST therefore carries a unique `?_=`
   cache-buster, which makes every request a cache miss.

`httpx` is used rather than `requests`: `requests` follows the `301`
by default and `allow_redirects=False` has to be threaded through every
call site. `httpx` is already a dependency of this project through
FastAPI.

---

## 2. What this means for the plan

| Plan item | Status after the spike |
|---|---|
| **Rewrite #1 — `ingestion/dre.py` replaces `eur_lex.py`** | **Confirmed, and cheaper than feared.** The parser is ~150 lines against a regular markup, not a per-era CSS archaeology project. The transport is the new work, not the parsing. |
| **Rewrite #2 — annex/table chunking** | **Premise wrong.** `TextoFormatado` contains **no `<table>` elements and no `<img>`** — verified across 10 diplomas. The numeric tables (emission-limit values, the REI's licensing thresholds) are **not in the HTML at all**; they exist only in the PDF. See §4. |
| **Rewrite #3 — manifest-driven fetcher** | **Confirmed and simplified.** The manifest needs `tipo` + `key` (both derivable from the public `/dr/detalhe/<tipo>/<key>` URL) plus the ELI and PDF URLs, which the API returns. Implemented as `scripts/fetch_corpus.py`. |
| **Rewrite #4 — `eval.yml` corpus steps** | **Confirmed.** The CI step is now one manifest-driven command instead of a per-diploma inline loop. |
| **Risk 1 — DRE availability/quality** | **Downgraded.** The content is reliably available, machine-readable, and free of the markup variety the plan feared. The real risk moved to §4 (tables) and §5 (consolidated text). |
| **Risk 2 — consolidated-text drift** | **Still open, and now sharper.** See §5. |
| **Risk 3 — DRE licensing** | **Still open, and now different.** See §6. |

---

## 3. Verified corpus

Ten diplomas were fetched and parsed end-to-end. The manifest
(`corpora.yaml`) currently carries six of them:

| Diploma | Published | Units | Articles | Annexes |
|---|---|---|---|---|
| DL 127/2013 (REI, licenciamento) | 2013-08-30 | 183 | 120 | 8 |
| DL 150/2015 (Seveso III) | 2015-08-05 | 61 | 54 | 6 |
| DL 102-D/2020 (RGGR, resíduos) | 2020-12-10 | 433 | 358 | 42 |
| DL 151-B/2013 (AIA) | 2013-10-31 | 59 | 52 | 6 |
| DL 226-A/2007 (águas) | 2007-05-31 | 102 | 98 | 3 |
| DL 56/2012 (orgânica APA) | 2012-03-12 | 19 | 18 | 0 |

Also verified and available: DL 127/2008 (PRTR), DL 89/2025 (altera o
REI), DL 121/2026 (altera o Seveso), DL 46/2008 (RCD).

The `tipo` + `key` pairs for all of these were recovered from
[`legal-citation-ledger.json`](./legal-citation-ledger.json), whose
`/dr/detalhe/...` URLs already contain them — the ledger was written
against the public site and needs no new research.

---

## 4. The tables problem is a PDF problem, not an HTML problem

This is the finding that most changes the plan.

The plan's Rewrite #2 assumes DRE serves HTML tables and that the
chunker must linearise them ("PT emission-limit tables … the
chemical-industry core content"). **There are no HTML tables.** Across
all ten fetched diplomas:

```
tags in TextoFormatado:  <p>, <a>, <br>      (no <table>, no <tr>, no <img>)
classes:                  paragraph-normal-text, paragraph-center,
                          paragraph-bold-center-14px,
                          paragraph-bold-center,
                          paragraph-title-bold-center-18px
```

What the REI's *Anexo I* actually contains in the HTML is prose and
lists ("1.1 Queima de combustíveis em instalações com uma potência
térmica nominal total igual ou superior a 50 MW"). The numeric tables
— emission-limit values by substance, by activity, by concentration —
are rendered only in the official PDF.

Consequences:

* A corpus built only from `TextoFormatado` **cannot answer
  emission-limit-value questions**. Those are precisely the questions a
  chemical-industry compliance tool is asked most often.
* The PDF is a separate, pinnable artefact: `URL_PDF` is a stable
  static file on `files.diariodarepublica.pt` (verified: HTTP 200,
  correct `application/pdf`, 325 KB for DL 150/2015), and the manifest
  already records it.
* Therefore the corpus needs a **PDF lane**: extract table-shaped
  content from the PDF (via `pdfplumber`/`camelot` or an OCR pass for
  the scanned issue pages), linearise each table into row-group chunks
  with repeated column headers, and keep the PDF's own sha256 in the
  corpus pins.

This is now the highest-value piece of remaining ingestion work, and
it is the piece the plan underestimated. It also has a licensing
consequence (§6).

---

## 5. Consolidated text is a different screen, not a flag

The plan's Risk 2 (consolidated-text drift) assumed the original and
the consolidated version are two flavours of the same response. They
are not — they are two different screens:

| | Original | Consolidated |
|---|---|---|
| Screen | `Legislacao_Conteudos/Conteudo_Detalhe` | `LegislacaoConsolidada/LegCons_Detalhe` |
| Action | `DataActionGetAllConteudoDetalheData` | `DataActionGetData` |
| Identifiers | `Tipo` + `Key` (e.g. `decreto-lei`, `150-2015-69951097`) | `DiplomaLegisId` + `DiplomaFragId` (e.g. `513863`, `70122774`) |
| Date selection | — | `DataSelecionada` (defaults to *today*) |

Two consequences:

1. The consolidated version can be pinned to a **specific
   consolidation date** by setting `DataSelecionada` — which is
   exactly what the plan wants for reproducibility. It is not yet
   implemented.
2. `DataSelecionada` defaults to the current date, so an un-pinned
   consolidated fetch silently returns *today's* consolidation. That is
   the drift the plan warned about, and it is one field away from being
   under control.

The consolidated screen also exposes the **amendment graph**
(`DataActionGetConsolidacaoByDiplomaFrag`,
`AlteracoesTimelineByDiplomaLegisId`), which is the mechanism the
methodology plan wanted for `amended_by` / `revoked_by` payload
fields. The DRE's own consolidated text carries the legal caveat that
it has *"texto consolidado sem valor legal"* — the generation layer
must disclose that, per the methodology plan §3.

---

## 6. Licensing: the picture changed, the question did not

The plan's Risk 3 assumed the reuse question was about *bulk scraping
the live site*. That risk has been defused by the transport change:
the fork now reads the same structured endpoint the official site
reads, one document at a time, with an identifying User-Agent. It is
no longer scraping.

Two questions remain open and are **not** engineering questions:

1. **Does the DRE's open-data regime permit redistribution of the
   extracted text in a commercial index?** The plan's own framing
   ("Lei 26/2016 lineage", "the PT open-data regime") still stands,
   but it now applies to *derived text* rather than to scraped HTML.
2. **Do the INCM-produced PDFs carry separate terms?** This became
   more important with §4: if the ELV tables have to come from the
   PDF, the PDF is on the critical path and its terms are no longer
   optional.

Neither question blocks Phases 0–4. Both must be answered before a
customer-facing index ships.

---

## 7. Revised risk register

| Risk | Before | After | Action |
|---|---|---|---|
| DRE corpus availability | High | **Low** | Closed. Machine-readable, reliable, verified across 10 diplomas. |
| HTML markup variability | High | **Low** | Closed. Five classes, one shape. |
| Table/ELV content | Medium (chunking work) | **High** (new lane) | New: PDF extraction lane. |
| Consolidated-text drift | Medium | **Medium** (now pinned-able) | Implement `DataSelecionada` pinning. |
| Licensing | Medium | **Medium** (different question) | Legal review before shipping. |
| Screenservice fragility | — | **New, Low–Medium** | Undocumented internal API; a DRE deploy could move it. The client reads the live bundle for its tokens, and the manifest pins the corpus. Add a smoke test that fails loudly when the contract changes. |
| Embedder swap regression | Medium | Unchanged | Dim asserted against model metadata. |

---

## 8. What is implemented right now

```
corpora.yaml                                 6 diplomas, manifest-driven
src/pt_amb_rag/config.py                     corpus + language + embedder config
src/pt_amb_rag/ingestion/dre.py              DRE client + structural parser
src/pt_amb_rag/ingestion/chunker.py          PT header grammar
src/pt_amb_rag/generation/prompts.py         PT prompt + PT citation grammar
src/pt_amb_rag/eval/judge.py                 PT rubric
src/pt_amb_rag/eval/corpus_pins.py           generalised to N diplomas
scripts/fetch_corpus.py                      manifest-driven fetcher + sha256
tests/ingestion/test_dre.py                  parser tests on a committed fixture
```

`ruff`, `ruff format`, `pyright` and `pytest` (245 tests) are green.

## 9. Recommended next steps, in order

1. **PDF lane** (§4) — table extraction and row-group chunking. Without
   it the corpus cannot answer the questions that matter most.
2. **Consolidated text with a pinned date** (§5) — implement
   `DataSelecionada` and record `consolidated_at` in the chunk payload,
   plus the *"sem valor legal"* disclosure in the prompt.
3. **Golden set** — the eval loop is the centrepiece of the upstream
   design and it is still empty here. The manifest, the parser and the
   pins are ready for it.
4. **Amendment graph** — `amended_by` / `revoked_by` payload fields
   from the consolidated screen.
5. **Hybrid retrieval** (BM25-PT + dense + RRF) — after the golden set
   exists, so the delta is measurable.
