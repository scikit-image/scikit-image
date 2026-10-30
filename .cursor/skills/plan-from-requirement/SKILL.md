---
name: plan-from-requirement
description: >-
  Read a Notion requirement and write a technical spec plus small user stories
  back to Notion, with a human gate, when the user types /plan-from-requirement.
disable-model-invocation: true
---

# Plan from requirement

`/plan-from-requirement <name of a Notion requirement page>` is for PMs and tech leads. It reads a requirement from Notion and writes a technical spec plus small user stories back to Notion, with a human gate.

## Procedure

1. **Read the requirement** — Work only in the Notion workspace the human names. If none is named, ask. Read the requirement page with the Notion tool. It should have a problem, a requirement, a "done when", and an out-of-scope section. If something is missing, ask.

2. **Ground the plan in the repository** — Search the repository and list only files and functions you confirm, with path and line. Never invent. Skip any item that already has a row with the same Name on the board "Good first tasks".

3. **Draft a technical specification** — Include:
   - Goal
   - Scope and out of scope
   - Files likely to change
   - Approach
   - Test cases (normal and edge) and where each test lives, following `.cursor/rules/repo-layout.mdc`
   - Risks
   - Docs to update (docstrings, doc/source pages, `doc/source/release_notes/release_dev.rst`)

4. **Draft user stories** — Draft up to 3 user stories unless the human asks for more. Each story has a one-line story, a checkable Acceptance text, the file, and an estimated size. Each story must fit the small-change skill's first-task definition (`.cursor/skills/small-change/SKILL.md`: at most 3 files and about 100 changed lines, no public API change, no compiled code). List any that do not fit under "Needs a person" and do not add them to the board.

5. **STOP** — Show the spec and the stories in the chat and wait for the human to approve or change them. Write nothing to Notion before that.

6. **After approval** — Create the spec as a new page "Tech spec: <requirement title>" under the requirement page. Add the approved stories as rows in "Good first tasks" with Status "Proposed" (add that option if it is missing), File and Acceptance filled, and a link to the spec in each row's page body.

7. **Report what was created.** A human changes Proposed to To do.

## Hard stops for the human

| When | Why |
|------|-----|
| After the draft (step 5) | Agree the spec and stories before anything is written |

## Explicit don’ts

- Never change the requirement page.
- Never set a story to To do.
- Never delete or move pages.
- Never write outside the named workspace.
- Never edit repo files or run git commands.
