#!/usr/bin/env python3
"""Export benchmark-params.json (written by resolve-benchmark-params.py
and passed between jobs as an artifact) into $GITHUB_ENV, so the
benchmark step's environment doesn't have to be spelled out one `env:`
entry at a time in the workflow.

The file's keys are already the environment variable names to set.
"""

import json
import os
import secrets

PARAMS_FILE = "benchmark-params.json"


def main() -> None:
    with open(PARAMS_FILE) as f:
        params = json.load(f)

    # Some values (the PR base/head labels) derive from user-controlled
    # branch names, so use the heredoc form with an unguessable
    # delimiter rather than KEY=VALUE - a value can't then terminate
    # its own block and inject further variables.
    delimiter = f"__BENCHMARK_PARAM_{secrets.token_hex(16)}__"

    with open(os.environ["GITHUB_ENV"], "a") as f:
        for key, value in params.items():
            if delimiter in str(value):
                raise ValueError(f"{key} contains the heredoc delimiter")
            f.write(f"{key}<<{delimiter}\n{value}\n{delimiter}\n")


if __name__ == "__main__":
    main()
