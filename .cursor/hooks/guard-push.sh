#!/bin/sh
# Deny risky git pushes before the shell runs them.
cat | python3 -c '
import json, shlex, sys
cmd = json.load(sys.stdin).get("command") or ""
try: args = shlex.split(cmd)
except ValueError: args = cmd.split()
def deny(user, agent):
    json.dump({"permission": "deny", "user_message": user, "agent_message": agent}, sys.stdout)
    raise SystemExit
stops = {"&&", "||", ";", "|"}
i = 0
while i < len(args) - 1:
    tok = args[i]
    if not ((tok == "git" or tok.endswith("/git")) and args[i + 1] == "push"):
        i += 1
        continue
    rest, i = [], i + 2
    while i < len(args) and args[i] not in stops:
        if args[i] != "--": rest.append(args[i])
        i += 1
    flags = [a for a in rest if a.startswith("-")]
    pos = [a for a in rest if not a.startswith("-")]
    # (a) push to the remote named upstream
    if pos[:1] == ["upstream"]:
        deny("Push to the remote named upstream is blocked.", "Denied: this git push targets the remote upstream.")
    # (b) force push: --force, -f, or --force-with-lease
    for a in flags:
        base = a.split("=", 1)[0]
        if base in ("--force", "--force-with-lease") or (not base.startswith("--") and "f" in base[1:]):
            deny("Force push is blocked (--force, -f, or --force-with-lease).", "Denied: this git push is a force push.")
    # (c) push directly to main or master
    for ref in pos[1:]:
        dest = (ref[1:] if ref.startswith("+") else ref).split(":")[-1]
        if (dest[11:] if dest.startswith("refs/heads/") else dest) in ("main", "master"):
            deny("Direct push to main or master is blocked.", "Denied: this git push targets main or master.")
json.dump({"permission": "allow"}, sys.stdout)
'
