# Eyetor

Multi-agent AI system based on Anthropic's agent patterns. Runs as a background service with simultaneous CLI and Telegram channels, tool use (shell, filesystem, browser, web search), and persistent memory across conversations.

## Requirements

- Python 3.11+
- A running LLM backend: [llama.cpp server](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.com), or an [OpenRouter](https://openrouter.ai) API key

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

**Telegram bot commands:** `/start`, `/reset`, `/skills`, `/help`

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

### google-workspace setup

Requires a Google Cloud project with the Calendar, Gmail, and Tasks APIs enabled.

1. Create OAuth 2.0 credentials (type: **Desktop App**) in [Google Cloud Console](https://console.cloud.google.com/) and download `credentials.json`
2. Place it at `~/.eyetor/google_credentials.json` (or set `GOOGLE_CREDENTIALS_FILE` env var)
3. Install dependencies:
   ```bash
   pip install google-api-python-client google-auth-oauthlib google-auth-httplib2
   ```
4. On first use the agent will trigger a browser OAuth flow — the token is saved automatically for subsequent runs

## Providers

| Provider | Type | Notes |
|----------|------|-------|
| llama.cpp | `llamacpp` | Local inference server, OpenAI-compatible |
| Ollama | `ollama` | Local model runner, OpenAI-compatible |
| OpenRouter | `openrouter` | Cloud proxy, requires API key |

A fallback chain retries across providers on timeout or server errors:

```yaml
fallback:
  fallback_chain: [llamacpp, openrouter]
  retry_on: [timeout, connection_error, "500", "502", "503"]
```
