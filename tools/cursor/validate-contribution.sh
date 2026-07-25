#!/usr/bin/env bash
# Local pre-PR checks: pre-commit on changed files + spin test --test-modified.
# Mirrors the fast path CI uses on pull requests. See AGENTS.md § Verification.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/cursor/validate-contribution.sh [OPTIONS]

Run pre-commit on changed paths, contribution heuristics, and spin test --test-modified.

Options:
  --base-ref REF       Base branch for test-modified (default: main)
  --module PATH        Extra spin test path after --test-modified (repeatable)
  --skip-pre-commit    Skip pre-commit (local debugging only)
  --allow-no-tests     Skip src-change vs tests/ pairing check (document why)
  -h, --help           Show this help

Examples:
  ./tools/cursor/validate-contribution.sh
  ./tools/cursor/validate-contribution.sh --base-ref main --module tests/skimage2/filters/
EOF
}

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "error: not inside a git repository" >&2
  exit 1
}
cd "$ROOT"

BASE_REF="main"
SKIP_PRECOMMIT=0
ALLOW_NO_TESTS=0
EXTRA_MODULES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-ref)
      BASE_REF="${2:?missing value for --base-ref}"
      shift 2
      ;;
    --module)
      EXTRA_MODULES+=("${2:?missing value for --module}")
      shift 2
      ;;
    --skip-pre-commit)
      SKIP_PRECOMMIT=1
      shift
      ;;
    --allow-no-tests)
      ALLOW_NO_TESTS=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

resolve_base() {
  if git rev-parse --verify "$BASE_REF" >/dev/null 2>&1; then
    echo "$BASE_REF"
    return
  fi
  if git rev-parse --verify "origin/$BASE_REF" >/dev/null 2>&1; then
    echo "origin/$BASE_REF"
    return
  fi
  echo "error: cannot resolve base ref '$BASE_REF' (try --base-ref or fetch remotes)" >&2
  exit 1
}

BASE="$(resolve_base)"
MERGE_BASE="$(git merge-base HEAD "$BASE" 2>/dev/null)" || {
  echo "error: cannot find merge-base with $BASE" >&2
  exit 1
}

CHANGED=()
while IFS= read -r line; do
  [[ -n "$line" ]] && CHANGED+=("$line")
done < <(
  {
    git diff --name-only --diff-filter=ACMRTUXB "$MERGE_BASE"
    git diff --name-only --diff-filter=ACMRTUXB
    git diff --cached --name-only --diff-filter=ACMRTUXB
  } | sort -u
)

echo "==> validate-contribution (base: $BASE, merge-base: ${MERGE_BASE:0:12}…)"

if [[ ${#CHANGED[@]} -eq 0 ]]; then
  echo "==> no changed files detected vs $BASE (including working tree)"
else
  echo "==> changed files (${#CHANGED[@]}):"
  printf '    %s\n' "${CHANGED[@]}"
fi

# --- Heuristic: behavior change under src/ should touch tests/ ---
is_src_behavior_path() {
  local f="$1"
  case "$f" in
    src/_skimage2/*)
      case "$f" in
        *.py | *.pyx | *.pxd) return 0 ;;
        */meson.build) return 0 ;;
      esac
      ;;
    src/skimage/*)
      case "$f" in
        *.py) return 0 ;;
        */meson.build) return 0 ;;
      esac
      ;;
  esac
  return 1
}

is_test_path() {
  local f="$1"
  [[ "$f" == tests/skimage/* || "$f" == tests/skimage2/* ]]
}

SRC_BEHAVIOR_CHANGED=0
TEST_CHANGED=0
TODO_CHANGED=0
for f in "${CHANGED[@]}"; do
  if is_src_behavior_path "$f"; then
    SRC_BEHAVIOR_CHANGED=1
  fi
  if is_test_path "$f"; then
    TEST_CHANGED=1
  fi
  if [[ "$f" == TODO.txt ]]; then
    TODO_CHANGED=1
  fi
done

if [[ "$SRC_BEHAVIOR_CHANGED" -eq 1 && "$TEST_CHANGED" -eq 0 ]]; then
  if [[ "$ALLOW_NO_TESTS" -eq 0 ]]; then
    echo "error: src/ behavior files changed but no tests/skimage/ or tests/skimage2/ changes detected." >&2
    echo "       Add or update tests, or re-run with --allow-no-tests (only for non-behavior changes)." >&2
    exit 1
  fi
  echo "==> heuristic: src/ changed without tests/ (--allow-no-tests)"
elif [[ "$SRC_BEHAVIOR_CHANGED" -eq 1 ]]; then
  echo "==> heuristic: src/ and tests/ both changed (OK)"
fi

# --- Heuristic: new deprecation helpers in src/ should update TODO.txt ---
DEPREC_IN_SRC=0
if git diff "$MERGE_BASE" -- src/ 2>/dev/null | grep -E '^\+' | grep -qE \
  'deprecate_func|deprecate_parameter|warn_external'; then
  DEPREC_IN_SRC=1
fi

if [[ "$DEPREC_IN_SRC" -eq 1 && "$TODO_CHANGED" -eq 0 ]]; then
  echo "error: diff adds deprecate_func, deprecate_parameter, or warn_external under src/" >&2
  echo "       but TODO.txt is not in the changed files." >&2
  echo "       Add a removal reminder per CONTRIBUTING.md (deprecation cycle)." >&2
  exit 1
elif [[ "$DEPREC_IN_SRC" -eq 1 ]]; then
  echo "==> heuristic: deprecation-related src/ changes include TODO.txt (OK)"
fi

if [[ "$SKIP_PRECOMMIT" -eq 0 ]]; then
  EXISTING=()
  for f in "${CHANGED[@]}"; do
    [[ -f "$f" ]] && EXISTING+=("$f")
  done
  if [[ ${#EXISTING[@]} -eq 0 ]]; then
    echo "==> pre-commit: no existing changed files; running on all files"
    pre-commit run --all-files
  else
    echo "==> pre-commit run --files (${#EXISTING[@]} files)"
    pre-commit run --files "${EXISTING[@]}"
  fi
else
  echo "==> pre-commit skipped (--skip-pre-commit)"
fi

echo "==> spin test --test-modified --base-ref $BASE_REF"
spin test --test-modified --base-ref "$BASE_REF"

for mod in "${EXTRA_MODULES[@]}"; do
  echo "==> spin test -- $mod"
  spin test -- "$mod"
done

echo "==> validate-contribution: OK"
