#!/usr/bin/env python3
"""Fail if test files live under src/."""
import sys
from pathlib import Path

# Step 1: repo root is two levels above .github/scripts/.
root = Path(__file__).resolve().parents[2]
src = root / "src"
message = (
    "Test files belong in top-level tests/ (tests/skimage/ for "
    "the legacy API, tests/skimage2/ for skimage2 and _skimage2 internals), "
    "not under src/."
)

# Step 2: scan every file under src/.
violations = []
for path in src.rglob("*"):
    if not path.is_file():
        continue
    # Step 3: conftest.py is allowed; flag test modules and tests/ directories.
    name = path.name
    if name == "conftest.py":
        continue
    parts = path.relative_to(src).parts
    is_test_module = name.endswith("_test.py") or (
        name.startswith("test_") and name.endswith(".py")
    )
    if is_test_module or "tests" in parts[:-1]:
        violations.append(path.relative_to(root).as_posix())

# Step 4: one GitHub annotation per violation.
for rel in violations:
    print(f"::error file={rel}::{message}")

# Step 5: one-line summary, then fail when src/ holds any test file.
print(f"{len(violations)} test file(s) under src/.")
sys.exit(1 if violations else 0)
