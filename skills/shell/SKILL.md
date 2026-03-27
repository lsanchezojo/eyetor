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
1. Run `scripts/run.py --cmd "<command>" [--cwd "<directory>"] [--timeout N]`
2. Returns JSON: `{"stdout": "...", "stderr": "...", "exit_code": 0}`

## Examples

Run a Python script:
```
scripts/run.py --cmd "python myscript.py" --cwd "/home/user/project"
```

Check git status:
```
scripts/run.py --cmd "git status"
```

Install npm packages:
```
scripts/run.py --cmd "npm install" --cwd "/home/user/myapp" --timeout 120
```

List running processes:
```
scripts/run.py --cmd "ps aux"        # Linux/macOS
scripts/run.py --cmd "tasklist"      # Windows
```

## Notes
- Default working directory is the current directory
- Default timeout: 30 seconds (override with --timeout)
- On Windows, commands run inside PowerShell (`powershell -Command "..."`)
- On Linux/macOS, commands run inside bash (`bash -c "..."`)
- For multi-line scripts, use a temp file and run it
- Commands that require interactive input will hang — avoid them
