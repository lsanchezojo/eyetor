# Eyetor

Multi-agent AI system based on Anthropic's agent patterns. Runs as a background service with simultaneous CLI and Telegram channels, tool use (shell, filesystem, browser, web search), and persistent memory across conversations.

## Requirements

- Python 3.11+
- A running LLM backend: [llama.cpp server](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.com), [OpenRouter](https://openrouter.ai), or [Google Gemini](https://ai.google.dev/) API key

## Installation

Eyetor se instala en un entorno virtual (`venv`) dentro del propio repositorio. Esto aísla sus dependencias y permite que el servicio systemd apunte a un intérprete fijo.

```bash
# 1. Clone and enter the repo
git clone <repo>
cd eyetor

# 2. Copy and configure environment
cp .env.example .env
# (edit .env with your tokens / API keys)

# 3. Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate            # bash/zsh
# source .venv/bin/activate.fish     # fish

# 4. Upgrade pip tooling (recommended)
pip install --upgrade pip wheel

# 5. Install Eyetor in editable mode + Telegram support
pip install -e ".[telegram]"

# 6. First-time host setup
eyetor setup
```

After installation the `eyetor` command is available at `.venv/bin/eyetor` (and on `PATH` while the venv is activated).

`eyetor setup` creates `~/.eyetor/host.json` with the detected operating system,
architecture, available package managers, and the preferred install command. The
agent loads this profile at startup so it does not assume the wrong OS package
manager. Regenerate it after moving the agent to another machine or installing a
new package manager:

```bash
eyetor setup --refresh-host
```

### Autonomous system package installs

By default Eyetor does not have `sudo` permissions. To let the agent install
system packages autonomously, install the restricted helper once:

```bash
sudo .venv/bin/eyetor setup --install-helper --service-user $USER
systemctl --user restart eyetor
```

This creates a root-owned `/usr/local/sbin/eyetor-install-tool` and a narrow
sudoers rule that only allows that helper. The agent uses the `install_package`
tool instead of arbitrary `sudo` commands. The helper validates package names
and automatically chooses the appropriate OS-specific install strategy for the
detected system; arbitrary shell commands are not enabled through this helper.

**Dependencies:**
- Core: `httpx`, `pyyaml`, `pydantic`, `rich`, `click`, `apscheduler`
- Optional (telegram): `aiogram`
- Optional (voice): `faster-whisper` — local speech-to-text for voice notes
- Optional (knowledge): `pypdf`, `python-docx`, `openpyxl`, `python-pptx`
- Optional (knowledge-vector): `fastembed`, `sqlite-vec`

To install knowledge-base extras too:

```bash
pip install -e ".[telegram,knowledge-full]"
```

## Development

Editable install (`-e`) means changes to `src/eyetor/` take effect on the next process start without reinstalling. Reinstall only when:

- You create new modules
- You change package/module structure (entry points, etc.)
- You pull updates that add new dependencies

```bash
source .venv/bin/activate
pip install -e ".[telegram]"
```

If the service is running, restart it to pick up the new code:

```bash
systemctl --user restart eyetor
```

## Configuration

```bash
cp .env.example .env
```

**.env**
```
TELEGRAM_BOT_TOKEN=     # from @BotFather on Telegram
TELEGRAM_ALLOWED_USER=  # your numeric chat_id (get it from @userinfobot)
OPENROUTER_API_KEY=     # only if using OpenRouter provider
GEMINI_API_KEY=         # only if using Gemini provider (LLM and/or images)
```

Main config: `config/default.yaml`. Key sections:

```yaml
providers:
  llamacpp:
    type: llamacpp
    base_url: http://localhost:8080/v1
    model: default
    temperature: 0.6

fallback:
  fallback_chain: [llamacpp, openrouter]   # first entry is primary, later ones are tried on retryable failures

channels:
  cli:
    host_tools: true      # enable shell/filesystem/browser/web-search
  telegram:
    enabled: true
    bot_token: ${TELEGRAM_BOT_TOKEN}
    auth:
      enabled: true
      allowed_users:
        - ${TELEGRAM_ALLOWED_USER}
      allowed_chats: []           # group chat_ids where any member may talk to the bot

memory_db_path: ~/.eyetor/memory.db

# Per-day, per-chat group conversation archive (see "Group chats" below)
chatlog_enabled: true
chatlog_db_path: ~/.eyetor/chatlog.db
chatlog_retention_days: 0         # 0 = keep forever
```

## Usage

```bash
# Start the agent (CLI if interactive, Telegram if configured)
eyetor start

# Start without host tools
eyetor start --no-host-tools

# One-shot query
eyetor run "what is the current directory?"

# List available skills
eyetor skills list

# Test provider connection
eyetor providers test
```

When run from a terminal, `eyetor start` opens an interactive CLI session. If Telegram is enabled in config, the bot starts simultaneously in the same process. When run without a terminal (e.g. as a systemd service), only the configured background channels start.

**CLI commands:** `/reset`, `/history`, `/skills`, `/agents`, `/tools`, `/model`, `/help`, `/exit`

**Telegram bot commands:** `/start`, `/reset`, `/skills`, `/agents`, `/tools`, `/model`, `/tasks`, `/usage`, `/help` + any commands declared by skills (see [Skill commands](#skill-commands))

### Group chats (Telegram)

The bot works both in direct messages and in group chats. In a group it behaves like a third participant:

- **Reads everything, replies only when addressed.** Every message is archived for context, but the bot only responds when you **@mention** it or **reply** to one of its messages — so it doesn't interrupt the conversation between humans.
- **Shared, identity-aware history.** Messages are stored prefixed with the sender's name (`[Luis]: ...`), so the model knows who said what in a multi-party conversation.
- **Anyone in an allowed group can talk to it.** Authorization in groups is per-chat: add the group's numeric `chat_id` (negative, e.g. `-1001234567890`) to `auth.allowed_chats`. Direct messages still use `auth.allowed_users`.

**Out-of-context archive.** Because the local model has a small context window, the full group history is **not** loaded into context. Instead it's archived per day and per chat in `chatlog.db` and queried **on demand** via tools the model calls when it needs to recall something — or when a user asks (e.g. "¿de qué hablamos ayer sobre X?"):

- `chat_history_search` — full-text search this chat's history (optionally filtered by day)
- `chat_history_read_day` — read a full day's transcript
- `chat_history_list_days` — list which days have history

Each tool is scoped to the current chat, so one group can never read another's logs. Set `chatlog_retention_days` > 0 to auto-purge old messages.

**Setup (one-time):** to let the bot receive group messages that don't mention it (needed for the history), disable its *privacy mode* in **@BotFather** → `/setprivacy` → select the bot → **Disable**. Then add the group's `chat_id` to `allowed_chats`. (To find the `chat_id`, add the bot to the group and check the service logs, or use a chat-id helper bot.)

### Image descriptions (Telegram)

When a user sends an image to the bot, it can describe its contents automatically. Configure a vision-capable model in `config/default.yaml`:

```yaml
vision_provider: llamacpp        # provider name (must exist under providers:)
vision_model: ggml-org/gemma-4-E4B-it-GGUF
```

Any OpenAI-compatible vision model works (Gemini, llama.cpp with a multimodal model, etc.). The description is injected into the conversation context before the user's message, so the agent can reason about the image normally.

### Voice messages (Telegram)

Send a voice (or audio) note and the bot transcribes it, echoes the text back as
`🎤 …`, and answers it like any typed message — handy for when you can't type.

**Enable it** by installing the `voice` extra (local, no server, fully private):

```bash
pip install -e ".[voice]"
```

Transcription is configured under `transcription:` in your config (defaults shown):

```yaml
transcription:
  enabled: true
  backend: local      # local = faster-whisper in-process | api = OpenAI-compatible endpoint
  model: medium       # small | medium | large-v3 (quality vs. speed)
  device: cpu         # CTranslate2 only accelerates on CUDA
  compute_type: int8  # fast and light on CPU
  language: es        # empty/null = autodetect
```

The first voice note downloads the chosen model (`medium` ≈ 1.5 GB) to the
HuggingFace cache; afterwards it's loaded from disk.

**Remote backend (optional).** Set `backend: api` (or just provide the env vars
below) to post the audio to an OpenAI-compatible `/v1/audio/transcriptions`
endpoint instead — e.g. a local whisper.cpp/Whisper server or the OpenAI API:

```
WHISPER_BASE_URL=http://localhost:8000   # local whisper server (or set base_url in config)
OPENAI_API_KEY=sk-...                    # OpenAI Whisper API (or set api_key in config)
```

## Deploying as a systemd user service

This is the recommended way to run Eyetor permanently: a **systemd user service** that points to the venv-installed binary, auto-starts on every reboot and restarts on failure. The service runs `eyetor start` without a tty, so only Telegram (or other non-interactive channels) start.

> **Prerequisite:** the venv must already be created, Eyetor installed, and `eyetor setup` run once (see [Installation](#installation)).

### 1. Create the service unit

Save the following file as `~/.config/systemd/user/eyetor.service`. Replace `/home/<user>/dev/workspace/eyetor` with the absolute path to **your** clone.

```ini
[Unit]
Description=Eyetor Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/<user>/dev/workspace/eyetor
EnvironmentFile=/home/<user>/dev/workspace/eyetor/.env
ExecStart=/home/<user>/dev/workspace/eyetor/.venv/bin/eyetor start
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=default.target
```

Key points:

- `ExecStart` points to `.venv/bin/eyetor`, the entry point installed inside the project's virtualenv — no need to activate the venv from systemd.
- `EnvironmentFile` loads variables from `.env` (Telegram token, API keys, etc.). Make sure `.env` exists at that path.
- `WorkingDirectory` is the project root so relative paths in `config/default.yaml` (e.g. `./skills`, `./plugins`) resolve correctly.

### 2. Reload, enable and start

```bash
systemctl --user daemon-reload
systemctl --user enable --now eyetor.service
```

`enable --now` both enables the unit (so it starts on every login) and starts it immediately.

### 3. Persist across logouts and reboots (lingering)

By default, user services stop when the user logs out and only start on login. To make Eyetor start on every boot **without requiring a login session**, enable lingering for your user (this is the only step that needs `sudo`):

```bash
sudo loginctl enable-linger $USER
```

Verify:

```bash
loginctl show-user $USER | grep Linger      # → Linger=yes
```

### 4. Verify it's running

```bash
systemctl --user status eyetor              # should show: active (running)
journalctl --user -u eyetor -n 30           # last 30 log lines
```

### Useful commands

```bash
journalctl --user -u eyetor -f          # live logs (follow)
systemctl --user status eyetor          # current status
systemctl --user restart eyetor         # restart after code/config changes
systemctl --user stop eyetor            # stop (will not restart until reboot/login)
systemctl --user disable --now eyetor   # stop and disable auto-start
```

### Updating a running deployment

```bash
cd /path/to/eyetor
git pull
source .venv/bin/activate
pip install -e ".[telegram]"            # only needed if dependencies changed
systemctl --user restart eyetor
```

### Uninstalling the service

```bash
systemctl --user disable --now eyetor
rm ~/.config/systemd/user/eyetor.service
systemctl --user daemon-reload
sudo loginctl disable-linger $USER      # optional
```

## Scheduled tasks

The agent can create, list, and cancel tasks that run automatically on a schedule. Tasks are persisted in SQLite (`~/.eyetor/scheduler.db`) and survive restarts.

### Managing tasks

Ask the agent in natural language:

```
"Remind me to drink water every hour"
"Every weekday at 8am summarise my unread emails and send here"
"Run a silent cleanup script every night at 2am"
"List my scheduled tasks"
"Cancel the task with id abc123"
```

The agent uses `schedule_task`, `list_tasks`, and `cancel_task` tools internally.

### Schedule format

| Format | Example | Meaning |
|--------|---------|---------|
| Cron (5 fields) | `0 9 * * *` | Every day at 09:00 |
| Cron (5 fields) | `0 8 * * 1` | Every Monday at 08:00 |
| Interval | `every 30m` | Every 30 minutes |
| Interval | `every 2h` | Every 2 hours |
| Interval | `every 1d` | Every day |

### Notification modes

| Mode | Behaviour |
|------|-----------|
| `telegram` (default) | Result sent to the Telegram chat that created the task |
| `log` | Result appended to `~/.eyetor/scheduler.log` (or a custom path) |
| `none` | Silent — task runs but no output is delivered |

### `/tasks` command

Use `/tasks` in Telegram at any time to list all scheduled tasks with their next run time and notification mode.

### Config

```yaml
scheduler:
  enabled: true
  db_path: ~/.eyetor/scheduler.db
  default_timezone: Europe/Madrid
```

## Interactive commands

### `/tools`

Lists all registered tools with descriptions — includes skill tools, MCP tools, and built-in tools (memory, scheduler, image generation). Available in both CLI and Telegram.

### `/model`

Switch provider and model on the fly without restarting:

```
/model                          → lists available providers and current selection
/model llamacpp                 → switch to llamacpp with its default model
/model openrouter mistral/...   → switch to openrouter with a specific model
```

Available in both CLI and Telegram.

## Session persistence

Optional JSONL persistence keeps conversation history across restarts. Enable in config:

```yaml
sessions:
  persist: true
  dir: ~/.eyetor/sessions
  max_messages: 200
```

Each session is stored as `<session_id>.jsonl` with one JSON message per line. History is loaded automatically when a session resumes and rotated when it exceeds `max_messages`.

## Conversation compaction

When using session persistence with long conversations, the context window can fill up. Eyetor supports **two-phase compaction** to summarize old messages:

```yaml
sessions:
  persist: true
  compaction:
    enabled: true
    context_window: 128000      # tokens in your model's context window
    trigger_at_percent: 0.80   # trigger at 80% of context (102.4K tokens)
    tool_output_max_chars: 2000  # Phase 1: truncate long tool outputs
    keep_last_n_user_turns: 2    # Preserve last N user turns verbatim
    summary_max_percent: 0.05   # Max summary = 5% of context (≈16k chars)
    archive_dir: ~/.eyetor/sessions/archive  # optional: save pre-compacted history
```

### How it works

1. **Phase 1 (cheap)**: Truncate tool outputs exceeding `tool_output_max_chars`
2. **Phase 2 (LLM)**: If still over threshold, summarize old messages via LLM
3. **Verbatim tail**: Last `keep_last_n_user_turns` are always kept intact

The summary uses a structured prompt that preserves file paths, commands, errors, and exact values.

**Note**: Compaction calls use `provider._inner` to bypass usage tracking (they don't count against daily limits).

## Plugins

Plugins extend Eyetor's behavior via subprocess hooks that run before/after tool executions. Plugins live in directories listed under `plugins_dirs` (default: `./plugins`, `~/.eyetor/plugins`).

### Plugin format

Each plugin is a directory with a `plugin.json`:

```json
{
  "name": "my-plugin",
  "version": "1.0.0",
  "description": "What this plugin does",
  "permissions": ["network"],
  "hooks": {
    "pre_tool_use": "hooks/pre.py",
    "post_tool_use": "hooks/post.py",
    "post_tool_use_failure": "hooks/post_fail.py"
  },
  "lifecycle": {
    "init": "setup.sh",
    "shutdown": "cleanup.sh"
  }
}
```

### Hook protocol

Hooks receive context as environment variables:

| Variable | Available in | Description |
|----------|-------------|-------------|
| `HOOK_EVENT` | all | `pre_tool_use`, `post_tool_use`, `post_tool_use_failure` |
| `HOOK_TOOL_NAME` | all | Tool name |
| `HOOK_TOOL_INPUT` | all | JSON arguments |
| `HOOK_TOOL_RESULT` | post | Tool result |
| `HOOK_TOOL_ERROR` | failure | Error message |
| `HOOK_TOOL_DURATION_MS` | post, failure | Execution time in milliseconds |

**Pre-hook decisions** (stdout JSON):
- `{"decision": "allow"}` — proceed normally
- `{"decision": "deny", "reason": "..."}` — block tool execution
- `{"decision": "modify", "input": {...}}` — modify arguments
- `{"decision": "provide_result", "result": "..."}` — return cached/synthetic result without executing

### Built-in plugins

| Plugin | Description |
|--------|-------------|
| `telegram-alerts` | Sends Telegram notifications when tools fail or exceed a duration threshold. Configure `bot_token`, `chat_id`, and `slow_threshold_ms` in `config.json` |
| `result-cache` | Caches results of expensive tools (web-search, browser) in SQLite with per-tool TTL. Pre-hook returns cached results; post-hook saves new ones. Configure `cached_tools`, TTLs, and `max_result_size_bytes` in `config.json` |

### Config

```yaml
plugins_dirs:
  - ./plugins
  - ~/.eyetor/plugins
```

## Memory

The agent has persistent memory backed by SQLite (`~/.eyetor/memory.db`). It can save facts, preferences, and notes across sessions using the `remember` and `forget` tools. Saved memories are injected into the system prompt at the start of every conversation.

In Telegram, memory is scoped per chat (`telegram-<chat_id>`): a direct chat has its own space, and a group has a single shared one. In CLI, memory is shared under the `cli` session. Group chats additionally keep a separate, out-of-context conversation archive (see [Group chats](#group-chats-telegram)).

## Skills

Skills live in `skills/` following the [agentskills.io](https://agentskills.io) format: a `SKILL.md` with YAML frontmatter and a `scripts/` subdirectory. Built-in skills:

| Skill | Description |
|-------|-------------|
| `shell` | Run shell commands on the host |
| `filesystem` | Read, write, list, search files |
| `browser` | Open URLs, fetch page content |
| `web-search` | Search the web via DuckDuckGo |
| `google-workspace` | Google Calendar, Gmail, and Tasks (requires setup) |

Use `/skills` in any channel (CLI or Telegram) to list active skills with their descriptions. New skills are picked up automatically at startup from all directories listed under `skills_dirs` in `config/default.yaml`.

### Skill commands

Skills can declare their own `/` commands for the Telegram bot by adding a `commands` block to their `SKILL.md` frontmatter. These are registered automatically at startup — no changes to the bot core needed.

Two action types are supported:

| Action | Behaviour |
|--------|-----------|
| `script` | Runs a skill script directly and sends its output to the chat |
| `prompt` | Sends a prompt to the agent session (supports `{args}` placeholder for user input) |

Example in `SKILL.md`:

```yaml
---
name: my-skill
description: "..."
commands:
  - name: mycommand
    description: "Does something useful"
    action: script
    script: my_script.py
    args: ["--format", "telegram"]
  - name: ask
    description: "Ask the agent a question about this skill"
    action: prompt
    prompt: "Using my-skill, answer: {args}"
---
```

**Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Command name without `/` (lowercase, alphanumeric + underscores) |
| `description` | yes | Shown in Telegram's command menu and `/help` |
| `action` | yes | `script` or `prompt` |
| `script` | if script | Script filename relative to `scripts/` |
| `args` | no | Default arguments passed to the script |
| `prompt` | if prompt | Prompt template; `{args}` is replaced with the user's input after the command |
| `parse_mode` | no | Telegram parse mode for script output (default: `HTML`) |

Reserved command names (`start`, `reset`, `skills`, `tools`, `model`, `tasks`, `usage`, `help`) cannot be overridden by skills.

### google-workspace setup

Requires a Google Cloud project with the Calendar, Gmail, and Tasks APIs enabled.

1. Create OAuth 2.0 credentials (type: **Desktop App**) in [Google Cloud Console](https://console.cloud.google.com/) and download `credentials.json`
2. Place it at `~/.eyetor/google_credentials.json` (or set `GOOGLE_CREDENTIALS_FILE` env var)
3. Install dependencies:
   ```bash
   pip install google-api-python-client google-auth-oauthlib google-auth-httplib2
   ```
4. On first use the agent will trigger a browser OAuth flow — the token is saved automatically for subsequent runs

## Subagents

Subagents are specialised worker agents the orchestrator can delegate to. Each one lives in its own Markdown file with YAML frontmatter (metadata) and a body that becomes its system prompt. The format is inspired by Anthropic's Agent SDK subagent definitions.

### File format

`agents/<name>.md`:

```markdown
---
name: researcher
description: Investigates topics thoroughly and returns verifiable facts.
temperature: 0.3      # optional — falls back to the orchestrator's temperature
# provider: openrouter # optional override — defaults to the orchestrator's provider
# model: anthropic/claude-...  # optional override
---

You are a meticulous researcher. For every subtask:

1. Identify the key claims to verify.
2. Return facts as a numbered list with sources where possible.
3. Mark unconfirmed claims explicitly as `(uncertain)`.
4. If you cannot answer, say so in one sentence — do not invent.
```

**Frontmatter fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Must match the filename stem (lowercase + `-`/`_`) |
| `description` | yes | One-line summary shown in `/agents` and in the orchestrator prompt |
| `provider` | no | Provider name from `providers:` — overrides the orchestrator's provider |
| `model` | no | Model identifier — overrides the orchestrator's model |
| `temperature` | no | Float — overrides the orchestrator's temperature |

The Markdown body (after the closing `---`) is the agent's full system prompt — write long, multi-paragraph instructions without needing to escape YAML.

### Discovery

Agents are scanned at startup from every directory listed in `agents_dirs`:

```yaml
agents_dirs:
  - ./agents             # versioned with the repo
  - ~/.eyetor/agents     # private / per-machine
```

Later directories override earlier ones on name collision. Use `/agents` in CLI or Telegram to list discovered agents with their descriptions.

### Activating delegation from chat

Set `auto_delegate: true` and list the workers you want available:

```yaml
orchestrator:
  auto_delegate: true
  protocol: auto
  workers:
    - researcher
    - sysadmin
```

When `auto_delegate` is on and at least one configured worker is found in `agents_dirs`, the main chat agent gets a `delegate` tool. The agent decides when to call it — typically when a subtask clearly matches one specialist (e.g. "research X", "fix this systemd unit"). Each call instantiates the subagent on the fly with its own system prompt, model, and temperature; the worker returns a single response which the main agent summarises into its reply.

Workers cannot see the main conversation — the main agent must include the necessary context in the `task` argument. This isolation is the point: each subagent has a focused prompt and a fresh context window.

**Unknown workers are skipped** at startup with a warning. If none of the configured workers are valid (or if `workers: []` is empty), the `delegate` tool is not registered and the main agent behaves normally.

### Built-in examples

| Agent | Description |
|-------|-------------|
| `researcher` | Investigates topics and returns verifiable facts with sources |
| `sysadmin` | Linux system administration — diagnostics, commands, destructive-op warnings |

## Usage tracking

Every LLM call is automatically tracked in SQLite (`~/.eyetor/tracking.db`). Recorded fields per call: timestamp, provider, model, prompt/completion tokens, cost, speed (tokens/second), finish reason, and session ID.

### Real cost from OpenRouter

When using OpenRouter, the agent records the **actual cost** reported by the API in each response (`usage.cost`). This is more accurate than the estimated pricing table. For providers that don't report costs (ollama, llamacpp), the built-in pricing table is used as a fallback.

### Viewing usage

**CLI:**

```bash
# Summary by provider/model (default: today)
eyetor usage
eyetor usage --period week --provider openrouter

# Individual call log
eyetor usage --detail
eyetor usage --detail -n 50
```

**Telegram:** Use `/usage` to see the current session stats and last 5 calls per model, plus today's summary.

### Daily limits

Set per-provider limits in `config/default.yaml`. Requests are blocked when a limit is reached, and the fallback chain tries the next available provider.

### Billing period

By default, the month period starts on day 1 at midnight. You can customize this to align with your billing cycle:

```yaml
tracking:
  db_path: ~/.eyetor/tracking.db
  month_start_day: 15    # Billing period starts on the 15th
  month_start_hour: 9    # at 9:00 AM
  limits:
    openrouter:
      daily_cost_usd: 10.0
      daily_tokens: 1000000
```

Period definitions:
- **day**: start of today (midnight)
- **week**: start of current week (Monday midnight)
- **month**: configurable (default: day 1 at midnight)

### Cost estimation

Costs are estimated from a built-in pricing table (`tracking/pricing.py`) covering OpenAI, Anthropic, Google, Meta, DeepSeek, and Qwen models. Local models (ollama, llamacpp) default to $0. The table can be extended as needed.

## Providers

| Provider | Type | Notes |
|----------|------|-------|
| llama.cpp | `llamacpp` | Local inference server, OpenAI-compatible |
| Ollama | `ollama` | Local model runner, OpenAI-compatible |
| OpenRouter | `openrouter` | Cloud proxy, requires API key |
| Google Gemini | `gemini` | Google AI, OpenAI-compatible endpoint. Supports both LLM and image generation |

A fallback chain retries across providers on timeout or server errors:

```yaml
fallback:
  fallback_chain: [llamacpp, openrouter]
  retry_on: [timeout, connection_error, "500", "502", "503"]
```

## Image generation

The agent can generate images via the `generate_image` tool. Multiple backends are supported:

| Provider | Type | Notes |
|----------|------|-------|
| OpenAI-compatible | `openai_compat` | Together AI, OpenAI DALL-E, any `/v1/images/generations` API |
| Google Gemini | `gemini` | Gemini Imagen API — can share config with the Gemini LLM provider |
| Automatic1111 | `automatic1111` | Stable Diffusion WebUI / Forge (`/sdapi/v1/txt2img`) |
| ComfyUI | `comfyui` | Workflow-based generation with custom templates |

### Configuration

Add an `image_providers` section to `config/default.yaml`:

```yaml
image_providers:
  gemini:
    type: gemini
    provider: gemini          # inherits config from providers.gemini
    model: gemini-2.0-flash-exp

default_image_provider: gemini
```

### Dual providers (LLM + image)

Providers like Google Gemini that support both text and image generation are configured once as a normal LLM provider, then referenced from `image_providers` to inherit connection details:

```yaml
providers:
  gemini:
    type: gemini
    base_url: https://generativelanguage.googleapis.com/v1beta
    api_key: ${GEMINI_API_KEY}
    model: gemini-2.0-flash

image_providers:
  gemini:
    type: gemini
    provider: gemini          # inherits base_url, api_key from providers.gemini
    model: gemini-2.0-flash-exp

fallback:
  fallback_chain: [llamacpp]              # primary LLM (add cloud fallbacks here)
default_image_provider: gemini            # images via Gemini
```

For OpenAI-compatible services (Together AI, OpenAI DALL-E, etc.):

```yaml
image_providers:
  together:
    type: openai_compat
    base_url: https://api.together.xyz/v1
    api_key: ${TOGETHER_API_KEY}
    model: black-forest-labs/FLUX.1-schnell
```

### Local Stable Diffusion

```yaml
image_providers:
  local_sd:
    type: automatic1111
    base_url: http://localhost:7860

  comfyui:
    type: comfyui
    base_url: http://localhost:8188
    workflow_template: ~/.eyetor/workflows/txt2img.json  # optional custom workflow
```

### How it works

When image generation is configured, the agent gets a `generate_image` tool. It generates the image, saves it to `~/.eyetor/generated_images/`, and includes an `[IMAGE:/path/to/file.png]` marker in the response. Channels render it accordingly:

- **Telegram:** sends the image as a photo with `answer_photo`
- **CLI:** displays the file path
