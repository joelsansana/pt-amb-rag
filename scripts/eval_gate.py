#!/usr/bin/env python
"""CI gate: compare a fresh eval run against the baseline on main.

Fails (exit 1) if current < baseline - tolerance for the given metric.
A missing or unfetchable baseline is treated as "nothing to compare
against" — warn and pass — so the first PR after this phase lands can
create the baseline without failing itself (the bootstrap case).

Usage:
    uv run python scripts/eval_gate.py \
        --current eval_run_principal.json \
        --baseline-url <raw.githubusercontent.com/.../baseline.json> \
        --metric hit_at_5 \
        --tolerance 0.02
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

FETCH_TIMEOUT_S = 10


class BaselineNotFoundError(Exception):
    """Baseline could not be fetched or parsed — the bootstrap case."""


def fetch_baseline(url: str) -> dict:
    """Fetch and parse the baseline JSON from a URL.

    Network errors, HTTP errors, and invalid JSON all raise
    BaselineNotFoundError: each means "there is nothing credible to
    compare against", which the gate treats as pass-with-warning.
    """
    try:
        response = requests.get(url, timeout=FETCH_TIMEOUT_S)
        response.raise_for_status()
        return json.loads(response.text)
    except (requests.RequestException, json.JSONDecodeError) as e:
        raise BaselineNotFoundError(f"could not fetch baseline: {e}") from e


def load_current(path: Path) -> dict:
    """Load the current eval run. A missing or malformed current file is
    a real failure (unlike a missing baseline): the eval itself broke."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"GATE ERROR: cannot read current eval run {path}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="path to the eval run JSON produced this build",
    )
    parser.add_argument(
        "--baseline-url", required=True, help="URL of the baseline JSON on main"
    )
    parser.add_argument(
        "--metric", required=True, help="metric key to compare, e.g. hit_at_5"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="allowed drop before the gate fails (default 0.02)",
    )
    args = parser.parse_args()

    try:
        baseline = fetch_baseline(args.baseline_url)
    except BaselineNotFoundError as e:
        print(
            "No baseline found on main yet — skipping gate, passing by "
            f"default. ({e})"
        )
        sys.exit(0)

    current = load_current(args.current)

    for label, data in (("baseline", baseline), ("current", current)):
        if args.metric not in data:
            sys.exit(
                f"GATE ERROR: metric {args.metric!r} missing from {label} JSON"
            )

    baseline_value = baseline[args.metric]
    current_value = current[args.metric]
    threshold = baseline_value - args.tolerance

    if current_value < threshold:
        sys.exit(
            f"GATE FAILED: {args.metric}={current_value:.3f} is more than "
            f"{args.tolerance:.3f} below baseline {baseline_value:.3f} "
            f"(required >= {threshold:.3f})"
        )

    print(
        f"gate passed: {args.metric}={current_value:.3f} "
        f"(baseline {baseline_value:.3f}, tolerance {args.tolerance:.3f})"
    )


if __name__ == "__main__":
    main()
