#!/bin/sh
cat | python3 -c '
import json, re, shlex, sys

def deny():
    msg = (
        "Only this form is allowed: git push [-u|--set-upstream|--dry-run] origin <branch>, "
        "where <branch> is not main, master, or HEAD and uses only letters, digits, dot, "
        "underscore, dash, and slash (no leading dash, +, :, or refs/)."
    )
    json.dump({"permission": "deny", "user_message": msg, "agent_message": msg}, sys.stdout)
    raise SystemExit

try:
    args = shlex.split(json.load(sys.stdin).get("command") or "")
except (ValueError, json.JSONDecodeError):
    deny()

# (1) exactly git push, then at most one of -u, --set-upstream, or --dry-run
if (
    len(args) == 5
    and args[:2] == ["git", "push"]
    and args[2] in ("-u", "--set-upstream", "--dry-run")
):
    remote, branch = args[3], args[4]
elif len(args) == 4 and args[:2] == ["git", "push"]:
    remote, branch = args[2], args[3]
else:
    deny()
# (2) remote is origin
if remote != "origin":
    deny()
# (3) one branch: [A-Za-z0-9._/-] but not main, master, HEAD, a flag, or refs/
if (
    branch in ("main", "master", "HEAD")
    or "refs/" in branch
    or re.fullmatch(r"[A-Za-z0-9._/][A-Za-z0-9._/-]*", branch) is None
):
    deny()
json.dump({"permission": "allow"}, sys.stdout)
'
