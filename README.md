# Eyetor

Multi-agent AI system based on Anthropic's agent patterns. Runs as a background service with simultaneous CLI and Telegram channels, tool use (shell, filesystem, browser, web search), and persistent memory across conversations.

## Requirements

- Python 3.11+
- A running LLM backend: [llama.cpp server](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.com), [OpenRouter](https://openrouter.ai), or [Google Gemini](https://ai.google.dev/) API key

## Installation

```bash
git clone <repo>
cd eyetor

# With Telegram support
pip install -e ".[telegram]"
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
default_provider: llamacpp

providers:
  llamacpp:
    type: llamacpp
    base_url: http://localhost:8080/v1
    model: default
    temperature: 0.6

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

memory_db_path: ~/.eyetor/memory.db
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

**CLI commands:** `/reset`, `/history`, `/skills`, `/help`, `/exit`

**Telegram bot commands:** `/start`, `/reset`, `/skills`, `/tasks`, `/usage`, `/help` + any commands declared by skills (see [Skill commands](#skill-commands))

### Voice messages (Telegram)

The bot transcribes voice and audio messages automatically. Transcription priority:

1. **Local Whisper server** — set `WHISPER_BASE_URL` in `.env` (e.g. `http://localhost:8000`), any OpenAI-compatible `/v1/audio/transcriptions` endpoint
2. **OpenAI Whisper API** — set `OPENAI_API_KEY` in `.env`
3. **Local faster-whisper** — install with `pip install faster-whisper`, no server needed

If none is configured, the bot replies with setup instructions.

```
WHISPER_BASE_URL=http://localhost:8000   # optional: local whisper server
OPENAI_API_KEY=sk-...                    # optional: OpenAI Whisper API
```

## Deploying as a systemd service

The service runs `eyetor start` without a tty, so only Telegram (or other non-interactive channels) start.

**1. Create the service unit**

Create `~/.config/systemd/user/eyetor.service`:

```ini
[Unit]
Description=Eyetor Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/path/to/eyetor
EnvironmentFile=/path/to/eyetor/.env
ExecStart=/home/<user>/.local/bin/eyetor start
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=default.target
```

Replace `/path/to/eyetor` with the absolute path to the project and `<user>` with your username.

**2. Enable and start**

```bash
systemctl --user daemon-reload
systemctl --user enable --now eyetor.service
```

**3. Persist across logouts**

```bash
loginctl enable-linger $USER
```

**Useful commands**

```bash
journalctl --user -u eyetor -f          # live logs
systemctl --user status eyetor          # status
systemctl --user restart eyetor         # restart after config changes
systemctl --user disable --now eyetor   # stop and disable
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

## Memory

The agent has persistent memory backed by SQLite (`~/.eyetor/memory.db`). It can save facts, preferences, and notes across sessions using the `remember` and `forget` tools. Saved memories are injected into the system prompt at the start of every conversation.

In Telegram, each user has an independent memory space. In CLI, memory is shared under the `cli` session.

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

Reserved command names (`start`, `reset`, `skills`, `tasks`, `usage`, `help`) cannot be overridden by skills.

### google-workspace setup

Requires a Google Cloud project with the Calendar, Gmail, and Tasks APIs enabled.

1. Create OAuth 2.0 credentials (type: **Desktop App**) in [Google Cloud Console](https://console.cloud.google.com/) and download `credentials.json`
2. Place it at `~/.eyetor/google_credentials.json` (or set `GOOGLE_CREDENTIALS_FILE` env var)
3. Install dependencies:
   ```bash
   pip install google-api-python-client google-auth-oauthlib google-auth-httplib2
   ```
4. On first use the agent will trigger a browser OAuth flow — the token is saved automatically for subsequent runs

## Usage tracking

Every LLM call is automatically tracked in SQLite (`~/.eyetor/tracking.db`). Recorded fields per call: timestamp, provider, model, prompt/completion tokens, estimated cost, speed (tokens/second), finish reason, and session ID.

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

**Telegram:** Use `/usage` to see the last 10 calls and today's summary.

### Daily limits

Set per-provider limits in `config/default.yaml`. Requests are blocked when a limit is reached, and the fallback chain tries the next available provider.

```yaml
tracking:
  db_path: ~/.eyetor/tracking.db
  limits:
    openrouter:
      daily_cost_usd: 10.0
      daily_tokens: 1000000
```

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

default_provider: llamacpp              # main LLM
default_image_provider: gemini          # images via Gemini
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
