# Eyetor

Multi-agent AI system based on Anthropic's agent patterns. Supports interactive CLI chat and Telegram bot, with tool use (shell, filesystem, browser, web search) backed by local or remote LLM providers.

## Requirements

- Python 3.11+
- A running LLM backend: [llama.cpp server](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.com), or an [OpenRouter](https://openrouter.ai) API key

## Installation

```bash
git clone <repo>
cd eyetor

# Base install
pip install -e .

# With Telegram bot support
pip install -e ".[telegram]"
```

## Configuration

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

**.env**
```
TELEGRAM_BOT_TOKEN=   # from @BotFather on Telegram
TELEGRAM_ALLOWED_USER=  # your numeric chat_id (get it from @userinfobot)
OPENROUTER_API_KEY=   # only if using OpenRouter provider
```

The main config file is `config/default.yaml`. The default provider is `llamacpp` pointing to `http://localhost:8080/v1`. Adjust `default_provider` and provider settings as needed:

```yaml
default_provider: llamacpp

providers:
  llamacpp:
    type: llamacpp
    base_url: http://localhost:8080/v1
    model: default
    temperature: 0.6

  openrouter:
    type: openrouter
    base_url: https://openrouter.ai/api/v1
    api_key: ${OPENROUTER_API_KEY}
    model: stepfun/step-3.5-flash:free
```

## Usage

```bash
# Interactive chat with host tools (shell, filesystem, browser, web-search)
eyetor chat

# Interactive chat without host access
eyetor chat --no-host-tools

# One-shot query
eyetor run "what is the current directory?"

# Telegram bot (foreground)
eyetor telegram

# List available skills
eyetor skills list

# Test provider connection
eyetor providers test
```

## Deploying as a systemd service (Telegram bot)

This runs the Telegram bot as a persistent background service under your user session.

**1. Create the service unit**

```bash
mkdir -p ~/.config/systemd/user
```

Create `~/.config/systemd/user/eyetor-telegram.service`:

```ini
[Unit]
Description=Eyetor Telegram Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/path/to/eyetor
EnvironmentFile=/path/to/eyetor/.env
ExecStart=/home/<user>/.local/bin/eyetor telegram
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=default.target
```

Replace `/path/to/eyetor` with the absolute path to the project and `<user>` with your username.

**2. Enable and start**

```bash
systemctl --user daemon-reload
systemctl --user enable --now eyetor-telegram.service
```

**3. Persist across logouts (optional)**

```bash
loginctl enable-linger $USER
```

**Useful commands**

```bash
# Live logs
journalctl --user -u eyetor-telegram -f

# Status
systemctl --user status eyetor-telegram

# Restart after config changes
systemctl --user restart eyetor-telegram

# Stop and disable
systemctl --user disable --now eyetor-telegram
```

## Skills

Skills live in `skills/` and follow the [agentskills.io](https://agentskills.io) format: a `SKILL.md` with YAML frontmatter and a `scripts/` subdirectory. Built-in skills:

| Skill | Description |
|-------|-------------|
| `shell` | Run shell commands on the host |
| `filesystem` | Read, write, list, search files |
| `browser` | Open URLs, fetch page content |
| `web-search` | Search the web via DuckDuckGo |

To add custom skills, place them in any directory listed under `skills_dirs` in `config/default.yaml`.

## Providers

| Provider | Type | Notes |
|----------|------|-------|
| llama.cpp | `llamacpp` | Local inference server, OpenAI-compatible |
| Ollama | `ollama` | Local model runner, OpenAI-compatible |
| OpenRouter | `openrouter` | Cloud proxy, requires API key |

A fallback chain can be configured so the agent retries across providers on timeout or server errors:

```yaml
fallback:
  fallback_chain: [llamacpp, openrouter]
  retry_on: [timeout, connection_error, "500", "502", "503"]
```
