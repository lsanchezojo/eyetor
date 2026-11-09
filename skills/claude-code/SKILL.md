---
name: claude-code
description: >-
  Delegate a coding or sysadmin task to Claude Code (a powerful autonomous coding
  agent) that runs in a git repo and edits files, runs commands, and commits.
  Use for real code changes, refactors, bug fixes, or system/ops tasks that need
  more than a single shell command. The 'args' string is: an optional repo name as
  the FIRST word, then the task in natural language, then optional flags. Flags:
  '--mode system' for ops/sysadmin framing, '--new' to start a fresh session
  instead of continuing the repo's last one. Examples: 'eyetor añade tests al
  módulo de memoria', 'odysseus arregla el login --new', 'reinicia nginx --mode
  system'. Claude Code keeps context per repo between calls, so follow-ups
  continue the same session automatically.
license: MIT
timeout: 1750
metadata:
  author: eyetor
  version: "1.0"
commands:
  - name: cc
    description: "Delega una tarea a Claude Code: /cc [repo] <tarea> [--mode system] [--new]"
    action: script
    script: cc.py
    parse_mode: ""         # texto plano: el message trae <...>/código que HTML no parsea
    reply_continue: true   # responder a un resultado continúa ese hilo exacto
---

# Claude Code

## When to use this skill
Use when the user wants an autonomous agent to actually **do** software or system
work in a repository, not just answer a question:

- Implement a feature, fix a bug, refactor, add tests.
- Run a multi-step coding workflow (edit → run → verify → commit).
- Sysadmin / ops tasks that need reasoning across several commands (`--mode system`).

For a single trivial command prefer the `shell` skill; for real coding tasks use this.

## How to invoke

The `args` string is parsed as:

```
[repo] <task in natural language> [--repo NAME|PATH] [--mode code|system] [--new]
```

- **repo** — optional. If the first word matches a configured repo name it selects
  that repo; otherwise the default repo is used. You can also pass `--repo <name>`
  or an absolute/`~` path.
- **task** — the natural-language instruction for Claude Code.
- **`--mode system`** — frame the task as sysadmin/ops work (diagnostics first,
  destructive ops flagged).
- **`--new`** — start a fresh Claude Code session. By default the skill **continues
  this conversation's own** Claude Code session for that repo (tracked in
  `~/.eyetor/claude_code_sessions.json`, keyed by chat+conversation), so follow-ups
  keep context and never bleed into another chat's — or a developer's — session in
  the same directory.
- If you omit the repo on a follow-up, the skill reuses the **last repo used in
  this conversation**, so it stays on the same project.
- **Reply-to-continue:** replying (Telegram reply) to a `/cc` result message
  continues **that message's exact thread** (its repo) automatically — even if you
  used another repo in between — with no need to retype `/cc`.
- **Conversations:** CC threads follow the eyetor conversation. `/reset` opens a
  new conversation → fresh CC threads; `/chat N` back to an old conversation
  restores its CC threads.

### Examples

```
eyetor añade tests unitarios al módulo de memoria
odysseus arregla el bug del login y haz commit --new
--repo eyetor documenta la función send() de session.py
libera espacio en disco de logs viejos --mode system
```

## Notes
- Claude Code runs **autonomously** (`--dangerously-skip-permissions`): it edits
  files and runs commands without asking. It is restricted to the directories in
  the skill's `config.json` allowlist.
- Configuration (binary path, repo shortcuts, default repo, allowlist, model,
  timeout) lives in `config.json` next to this skill — edit it there.
- The reply is a concise summary: Claude Code's final message, the list of
  uncommitted changes (`git status --short`), and run metadata (turns, time, cost,
  session id). It is **not** the full transcript.
- Cost is billed to the host's Claude subscription, separate from eyetor's own
  provider usage tracking.
