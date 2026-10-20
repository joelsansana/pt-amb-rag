# Contributing to `pt-amb-rag`

Thanks for your interest in this project. It's a portfolio-grade RAG system over Portuguese environmental legislation, and contributions that improve retrieval quality, evaluation rigour, or corpus coverage are very welcome.

## Quick links

- [Bug reports](#bug-reports)
- [Feature requests](#feature-requests)
- [Pull requests](#pull-requests)
- [Development setup](#development-setup)
- [Style and conventions](#style-and-conventions)
- [Code of conduct](#code-of-conduct)

---

## Bug reports

Open an issue using the **Bug report** template. Include:

1. **What you did** (exact command, query, or step)
2. **What you expected to happen**
3. **What actually happened** (full traceback, log excerpt, or screenshot)
4. **Your environment** (`uv --version`, `python --version`, OS)

## Feature requests

Open an issue using the **Feature request** template. Include:

1. **The problem you're solving**, not just the solution
2. **A worked example** showing the current behaviour and the desired behaviour
3. **Why this is in scope** (cite the README's "What NOT to build" section if you're pushing back against it)

For large changes, please **open the issue first** before writing code. This avoids wasted effort if the direction doesn't fit the project's scope.

## Pull requests

1. **Fork the repo** and create a feature branch (`git checkout -b feat/hybrid-reranker`)
2. **Make your change.** Add tests. Update docs.
3. **Run the full pre-commit gauntlet:**

   ```bash
   uv sync --all-extras
   uv run ruff check .
   uv run ruff format --check .
   uv run pyright src tests
   uv run pytest -q
   ```

4. **Open the PR** using the PR template. The CI gate (`.github/workflows/ci.yml`) must pass — that includes ruff lint, format check, pyright, and pytest.
5. **For retrieval-affecting changes:** if Phase 4 (eval) has landed, the eval CI gate (`eval.yml`) will run on your PR. Expect the bar: Hit@5 must not regress by more than 2 points vs. `main`. If your change legitimately lowers Hit@5 in exchange for a metric that matters more (e.g., lower latency at similar recall), explain it in the PR body — the regression gate is a strong default, not a hard rule.

### PR conventions

- **One concern per PR.** Don't bundle a corpus fix with a UI change.
- **Reference the issue** in the PR body ("Closes #42").
- **Public-API changes** belong in their own PR with a short migration note in the PR body.
- **Squash commits** on merge (the maintainer will do this; keep your branch history readable).

---

## Development setup

```bash
# Clone
git clone https://github.com/joelsansana/pt-amb-rag.git
cd pt-amb-rag

# Python + dependencies (uv-managed)
uv sync --all-extras

# (Optional) Run a quick test to confirm everything works
uv run pytest -q tests/test_smoke.py
```

You'll need Python 3.12 and `uv` installed. Docker is required from Phase 3 onwards. The integration tests (LLM-as-judge against MiniMax) need a `MINIMAX_API_KEY` in your environment and are skipped by default — opt in with `uv run pytest -m integration`.

## Style and conventions

- **Ruff** for lint + format (line length 100, target Python 3.12). Run `uv run ruff format .` before committing.
- **Pyright** for type checking (strict mode). Run `uv run pyright src tests`.
- **Pytest** for tests. One assertion concept per test; fixtures live in `tests/conftest.py`.
- **Module docstrings** on every public function. Type hints everywhere (use `from __future__ import annotations`).
- **No LangChain, no LlamaIndex.** Hand-rolled where reasonable. The `openai` Python client is the only LLM SDK.
- **No new top-level dependencies without discussion.** If a dependency is genuinely needed, open an issue first.

## Code of conduct

This is a small, focused project. Be kind, be specific in your feedback, and assume good faith. Disagreements about design should be argued with citations to the README or BUILD_PLAN, not volume.

---

*Thanks for helping make `pt-amb-rag` more useful and more honest.*
