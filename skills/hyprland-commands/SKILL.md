---
name: hyprland-commands
description: Controla el escritorio Hyprland desde el chat (encender pantallas, brillo y otras acciones del compositor). Ejecuta comandos deterministas vía hyprctl y utilidades del sistema.
license: MIT
compatibility: Python 3.11+ sobre Linux con Hyprland. Requiere `hyprctl` y `brightnessctl` en el PATH y la variable HYPRLAND_INSTANCE_SIGNATURE del compositor en ejecución.
metadata:
  author: eyetor
  version: "0.1"
timeout: 30
commands:
  - name: hypr_hello
    description: Despierta las pantallas (DPMS on) y restaura el brillo guardado.
    action: script
    script: hypr.py
    args:
      - hello
  - name: hypr_bye
    description: Apaga las pantallas (DPMS off).
    action: script
    script: hypr.py
    args:
      - bye
  - name: hypr_shot
    description: "Captura un workspace y lo envía al chat. Uso: /hypr_shot [número]; vacío = activo."
    action: script
    script: hypr.py
    args:
      - shot
  - name: hypr_lock
    description: Bloquea la sesión.
    action: script
    script: hypr.py
    args:
      - lock
  - name: hypr_status
    description: Resumen de ventana activa, monitores y workspaces.
    action: script
    script: hypr.py
    args:
      - status
  - name: hypr_notify
    description: "Muestra un aviso en la pantalla del equipo. Uso: /hypr_notify <texto>."
    action: script
    script: hypr.py
    args:
      - notify
  - name: hypr_volume
    description: "Controla el volumen. Uso: /hypr_volume up|down|mute|<porcentaje>."
    action: script
    script: hypr.py
    args:
      - volume
  - name: hypr_media
    description: "Controla la reproducción. Uso: /hypr_media play-pause|next|prev|stop."
    action: script
    script: hypr.py
    args:
      - media
---

# Hyprland Commands

## When to use this skill
Use when the user asks to control the Hyprland desktop / host remotely, e.g.:
- Wake up or turn off the monitors, restore/adjust brightness
- Take a screenshot of what's on screen
- Lock the session
- Check what's running (active window, monitors, workspaces)
- Show a notification on the host screen
- Control audio volume or media playback

## Scripts

`hypr.py` runs deterministic compositor/system actions and returns a short
status message. Each action is a subcommand:

```
hypr.py hello                # hyprctl dispatch dpms on && brightnessctl -r
hypr.py bye                  # hyprctl dispatch dpms off
hypr.py shot [workspace]     # grim screenshot of a workspace (default: active) -> image_path
hypr.py lock                 # loginctl lock-session (fallback: hyprlock)
hypr.py status               # active window + monitors + workspaces
hypr.py notify <text...>     # notify-send
hypr.py volume up|down|mute|<percent>   # wpctl / pactl
hypr.py media play-pause|next|prev|stop # playerctl
```

## Notes
- Requires `hyprctl` on the host PATH. Per action, also: `brightnessctl`
  (hello), `grim` (shot; `grimblast` as fallback for the no-arg whole screen),
  `loginctl`/`hyprlock` (lock), `notify-send`/libnotify (notify),
  `wpctl` or `pactl` (volume), `playerctl` (media). Missing tools are reported
  as an error.
- The script must run in the same session as the Hyprland compositor
  (it relies on HYPRLAND_INSTANCE_SIGNATURE being set in the environment).
- `shot` with a workspace number that is **not currently visible** switches to
  it momentarily (a brief flicker on the host screen), captures, and restores
  the previous workspace. A visible workspace is captured without switching.
- Output is JSON: `{"ok": true, "message": "..."}` on success (the `shot`
  action also includes `"image_path"`, which the channel sends as a photo),
  or `{"error": "..."}` on failure.
