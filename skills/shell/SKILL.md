---
name: shell
description: Execute shell commands on the host system and return their output. Use for running programs, scripts, package managers, git operations, build tools, or any terminal command.
license: MIT
compatibility: Python 3.11+. Uses PowerShell on Windows, bash on Linux/macOS.
metadata:
  author: eyetor
  version: "1.0"
---

# Shell

## When to use this skill
Use when the user asks to:
- Run a command or script (`npm install`, `git status`, `python script.py`)
- Check system state (`ps aux`, `df -h`, `systemctl status`)
- Execute build/test/deploy pipelines
- Automate repetitive terminal tasks
- Interact with CLI tools (docker, kubectl, ffmpeg, etc.)

## How to run a command

**Canonical form** — wrap the command in `--cmd "..."`:
```
--cmd "<command>" [--cwd "<directory>"] [--timeout N]
```

The whole command must live inside a single `--cmd "..."` quoted string
(including pipes, redirects, and arguments). Returns JSON:
`{"stdout": "...", "stderr": "...", "exit_code": 0}`.

## Examples

Get today's date:
```
--cmd "date +%Y-%m-%d"
```

Run a Python script:
```
--cmd "python myscript.py" --cwd "/home/user/project"
```

Check git status:
```
--cmd "git status"
```

Install npm packages:
```
--cmd "npm install" --cwd "/home/user/myapp" --timeout 120
```

List running processes:
```
--cmd "ps aux"        # Linux/macOS
--cmd "tasklist"      # Windows
```

## Notes
- Default working directory is the current directory
- Default timeout: 30 seconds (override with --timeout)
- On Windows, commands run inside PowerShell (`powershell -Command "..."`)
- On Linux/macOS, commands run inside bash (`bash -c "..."`)
- For multi-line scripts, use a temp file and run it
- Commands that require interactive input will hang — avoid them
