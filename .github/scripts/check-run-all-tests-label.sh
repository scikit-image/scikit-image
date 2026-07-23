#!/usr/bin/env bash
# Check if the current PR has the 'run-all-tests' label using the gh CLI.
# Writes test_modified=true/false to $GITHUB_OUTPUT.
#
# Required env vars:
#   PR_NUMBER: The pull request number (empty for non-PR events)
#   GH_TOKEN:  GitHub token with repo read access (passed as secrets.GITHUB_TOKEN)

set -e

if [[ -z "$PR_NUMBER" ]]; then
  # Not a pull request — run all tests
  echo "Not a pull request, run all tests"
  echo "test_modified=false" >> "$GITHUB_OUTPUT"
  exit 0
fi

# Fetch label state dynamically (not from workflow trigger context) so that
# re-running the workflow after adding the 'run-all-tests' label picks up
# the new label state correctly.
HAS_LABEL=$(gh pr view "$PR_NUMBER" --json labels \
  --jq '[.labels[].name] | contains(["run-all-tests"])')

if [[ "$HAS_LABEL" == "true" ]]; then
  echo "Run all tests"
  echo "test_modified=false" >> "$GITHUB_OUTPUT"
else
  echo "Run only modified tests"
  echo "test_modified=true" >> "$GITHUB_OUTPUT"
fi
