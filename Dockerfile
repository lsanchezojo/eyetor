FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (compatible with rootless Podman)
RUN useradd -m -u 1000 -s /bin/bash eyetor
ENV HOME=/home/eyetor

# Copy project files
COPY --chown=eyetor:eyetor pyproject.toml .
COPY --chown=eyetor:eyetor src/ src/
COPY --chown=eyetor:eyetor skills/ skills/
COPY --chown=eyetor:eyetor config/ config/

# Install Python dependencies (including optional telegram support)
RUN pip install --no-cache-dir -e ".[telegram]"

# Create data directory for SQLite databases and user skills
RUN mkdir -p /home/eyetor/.eyetor/skills && chown -R eyetor:eyetor /home/eyetor/.eyetor

USER eyetor

# Default command: show help (override in compose)
CMD ["eyetor", "--help"]
