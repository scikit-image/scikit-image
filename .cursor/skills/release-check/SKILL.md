---
name: release-check
description: >-
  Say whether one pull request can merge, and if not why, when the user types /release-check.
disable-model-invocation: true
---

# Release check

`/release-check <pull request number>` is for PM and DevOps. It answers whether this can merge, and if not, why. Read-only.

## Procedure

1. **Fork** — Run `git remote get-url origin`. `<fork>` is `owner/repo` from that URL (drop a trailing `.git`; keep the last two path parts). Never use `upstream`.

2. **Read status** — Run all four. `gh pr checks` may exit non-zero; still use its output. If any command fails, verdict is Blocked and the sentence names the command.
   - `gh pr checks <n> --repo <fork>`
   - `gh pr view <n> --repo <fork> --json isDraft,mergeable,reviewDecision,state`
   - `gh api repos/<fork>/rules/branches/main`
   - `gh api graphql -f query='query{repository(owner:"<owner>", name:"<repo>"){pullRequest(number:<n>){reviewThreads(first:50){nodes{isResolved isOutdated comments(first:1){nodes{author{login} body path}}}}}}}'`

   `<owner>` and `<repo>` come from the fork. An open finding is a thread where `isResolved` is false. Label each open finding's source from the first comment's text: "Cursor Bugbot" in the text means Bugbot, "Agentic Security Review" means the security reviewer, otherwise "other". Do not filter by author name. Use each open finding's title (the first line of the first comment if it has no separate title).

## Answer

- **Verdict:** `Ready` or `Blocked`.
- **Required checks and their state.** Required checks come from the ruleset on `main`. Their state comes from `gh pr checks`.
- **Open findings with titles.**
- **Draft:** whether it is a draft. Drafts get no Bugbot review.
- **One sentence** a non-engineer can read.

`Blocked` when any of these is true: `state` is not `OPEN`; `isDraft` is true; `mergeable` is not `MERGEABLE`; a required check is failing, pending, or missing; the ruleset on `main` is not satisfied (including `reviewDecision` when the ruleset requires review); an open finding remains (`isResolved` is false). Otherwise `Ready`. A resolved thread is not an open finding and must not make the verdict Blocked. When `Blocked`, the one sentence says why.

## Explicit don'ts

- Never merge.
- Never approve.
- Never comment.
- Never change settings.
