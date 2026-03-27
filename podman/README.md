# Vector — Despliegue con Podman

Hay dos formas de desplegar con Podman:

---

## Opción A: `podman compose` (desarrollo / Windows)

Compatible directamente con el `docker-compose.yml` del proyecto.

```bash
# Construir imagen
podman build -t vector:latest .

# Copiar y editar variables de entorno
cp .env.example .env

# Chat interactivo (sesión de terminal)
podman compose run --rm vector-chat

# Bot de Telegram (daemon)
podman compose --profile telegram up -d vector-telegram

# Ver logs
podman compose logs -f vector-telegram

# Parar
podman compose down
```

> En Windows con Podman Desktop se puede usar `podman compose` igual que `docker compose`.

---

## Opción B: Quadlet + systemd (Linux / WSL2 — recomendado para 24/7)

Quadlet es la forma nativa de Podman para integración con systemd.
Los servicios arrancan automáticamente con el sistema, se reinician solos y sus logs van a journald.

### 1. Construir imagen

```bash
podman build -t vector:latest .
```

### 2. Setup automático

```bash
bash podman/setup.sh
```

El script:
- Construye la imagen
- Copia los archivos `.container` y `.volume` a `~/.config/containers/systemd/`
- Recarga el daemon de systemd
- Imprime los comandos de activación

### 3. Activar servicios

```bash
# Bot Telegram (24/7)
systemctl --user enable --now vector-telegram.service

# Ver estado y logs
systemctl --user status vector-telegram
journalctl --user -u vector-telegram -f
```

### 4. Arranque automático tras reinicio / logout

```bash
loginctl enable-linger $(whoami)
```

Esto hace que los servicios de usuario sigan corriendo aunque no haya sesión activa.

---

## Chat interactivo con Podman

```bash
podman run -it --rm \
  -v vector-data:/home/vector/.vector \
  -v ./skills:/app/skills:ro \
  -v ./config:/app/config:ro \
  --env-file .env \
  vector:latest vector chat
```

---

## Comandos útiles

```bash
# Estado de los contenedores
podman ps

# Logs
podman logs -f vector-telegram

# Ejecutar comando dentro del contenedor
podman exec vector-telegram vector providers list
podman exec vector-telegram vector skills list

# Actualizar imagen y reiniciar servicio
podman build -t vector:latest .
systemctl --user restart vector-telegram.service

# Parar y eliminar servicio
systemctl --user disable --now vector-telegram.service
```

---

## Estructura de archivos Quadlet

```
~/.config/containers/systemd/
├── vector-data.volume          # Volumen persistente (SQLite)
├── vector-telegram.container   # Bot Telegram 24/7
└── vector-agent.container      # Daemon genérico 24/7
```

Los archivos `.container` son traducidos automáticamente por systemd-generator
a units `.service` — no necesitas crear los `.service` manualmente.
