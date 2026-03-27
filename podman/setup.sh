#!/usr/bin/env bash
# Eyetor — Podman setup script
# Run from the project root: bash podman/setup.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUADLET_DIR="${HOME}/.config/containers/systemd"
CONFIG_DIR="${HOME}/.config/eyetor"

echo "==> Building Eyetor image with Podman..."
podman build -t eyetor:latest "${PROJECT_DIR}"

echo "==> Creating config directory: ${CONFIG_DIR}"
mkdir -p "${CONFIG_DIR}"

# Copy .env if it doesn't exist
if [ ! -f "${CONFIG_DIR}/.env" ]; then
  if [ -f "${PROJECT_DIR}/.env" ]; then
    cp "${PROJECT_DIR}/.env" "${CONFIG_DIR}/.env"
    echo "    Copied .env to ${CONFIG_DIR}/.env"
  else
    cp "${PROJECT_DIR}/.env.example" "${CONFIG_DIR}/.env"
    echo "    Created ${CONFIG_DIR}/.env from .env.example"
    echo "    *** Edit ${CONFIG_DIR}/.env with your API keys before starting services ***"
  fi
fi

# Create symlink so eyetor finds the project config/skills
mkdir -p "${HOME}/eyetor"
[ -L "${HOME}/eyetor/skills" ] || ln -s "${PROJECT_DIR}/skills" "${HOME}/eyetor/skills"
[ -L "${HOME}/eyetor/config" ] || ln -s "${PROJECT_DIR}/config" "${HOME}/eyetor/config"

echo "==> Installing Quadlet units to ${QUADLET_DIR}..."
mkdir -p "${QUADLET_DIR}"
cp "${PROJECT_DIR}/podman/eyetor-data.volume"        "${QUADLET_DIR}/"
cp "${PROJECT_DIR}/podman/eyetor-telegram.container" "${QUADLET_DIR}/"
cp "${PROJECT_DIR}/podman/eyetor-agent.container"    "${QUADLET_DIR}/"

echo "==> Reloading systemd user daemon..."
systemctl --user daemon-reload

echo ""
echo "==> Setup complete. Available services:"
echo ""
echo "  eyetor-telegram (Telegram bot, 24/7):"
echo "    systemctl --user enable --now eyetor-telegram.service"
echo "    journalctl --user -u eyetor-telegram -f"
echo ""
echo "  eyetor-agent (daemon, 24/7):"
echo "    systemctl --user enable --now eyetor-agent.service"
echo ""
echo "  Interactive chat (not a daemon):"
echo "    podman run -it --rm -v eyetor-data:/home/eyetor/.eyetor \\"
echo "      -v ${PROJECT_DIR}/skills:/app/skills:ro \\"
echo "      -v ${PROJECT_DIR}/config:/app/config:ro \\"
echo "      --env-file ${CONFIG_DIR}/.env eyetor:latest eyetor chat"
echo ""
echo "  To start services at system boot (after logout):"
echo "    loginctl enable-linger \$(whoami)"
echo ""
