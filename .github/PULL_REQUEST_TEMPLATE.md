---
name: Pull request
about: Submit a change to pt-amb-rag
---

## What this PR does

<!-- One paragraph. What changes, why now, who benefits. -->

## Related issue

<!-- Link to the issue this closes, or "N/A." -->

Closes #

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation only

## Checklist

- [ ] I have read [`CONTRIBUTING.md`](./CONTRIBUTING.md)
- [ ] I have added tests for my change (or explained why none are needed)
- [ ] All tests pass locally: `uv run pytest -q`
- [ ] Lint passes: `uv run ruff check . && uv run ruff format --check .`
- [ ] Types pass: `uv run pyright src tests`
- [ ] I have updated the relevant docs (README, BUILD_PLAN, TUTORIAL) if my change affects user-facing behaviour

## Notes for the reviewer

<!-- Anything that needs special attention: trade-offs, scope decisions, edge cases you handled. Delete if empty. -->
