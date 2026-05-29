#!/usr/bin/env python3
"""Control Hyprland from the chat.

Usage:
  hypr.py hello              Wake screens (DPMS on) and restore saved brightness.
  hypr.py bye                Turn screens off (DPMS off).
  hypr.py shot               Take a screenshot and return its path.
  hypr.py lock               Lock the session.
  hypr.py status             Active window, monitors and workspaces summary.
  hypr.py notify <text...>   Show a desktop notification on the host screen.
  hypr.py volume [up|down|mute|<percent>]   Control audio volume.
  hypr.py media [play-pause|next|prev|stop] Control media playback.

Each subcommand runs a deterministic compositor/system action and prints a JSON
result: {"ok": true, "message": "..."} on success (optionally with
"image_path"), or {"error": "..."} on failure.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time


def _emit(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload.get("ok") else 1


def _run(cmd: list[str], timeout: float = 15.0) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def cmd_hello(_args: argparse.Namespace) -> int:
    """hyprctl dispatch dpms on && brightnessctl -r"""
    for tool in ("hyprctl", "brightnessctl"):
        if shutil.which(tool) is None:
            return _emit({"error": f"'{tool}' no está en el PATH"})

    code, _out, err = _run(["hyprctl", "dispatch", "dpms", "on"])
    if code != 0:
        return _emit({"error": f"hyprctl dispatch dpms on falló: {err or code}"})

    code, _out, err = _run(["brightnessctl", "-r"])
    if code != 0:
        return _emit({"error": f"brightnessctl -r falló: {err or code}"})

    return _emit({"ok": True, "message": "Pantallas encendidas y brillo restaurado."})


def cmd_bye(_args: argparse.Namespace) -> int:
    """hyprctl dispatch dpms off"""
    if shutil.which("hyprctl") is None:
        return _emit({"error": "'hyprctl' no está en el PATH"})

    code, _out, err = _run(["hyprctl", "dispatch", "dpms", "off"])
    if code != 0:
        return _emit({"error": f"hyprctl dispatch dpms off falló: {err or code}"})

    return _emit({"ok": True, "message": "Pantallas apagadas."})


def cmd_shot(_args: argparse.Namespace) -> int:
    """Screenshot with grimblast (preferred) or grim, return its path."""
    out_path = os.path.join(
        tempfile.gettempdir(), f"eyetor-shot-{int(time.time())}.png"
    )

    if shutil.which("grimblast") is not None:
        code, _out, err = _run(["grimblast", "save", "screen", out_path], timeout=20.0)
    elif shutil.which("grim") is not None:
        code, _out, err = _run(["grim", out_path], timeout=20.0)
    else:
        return _emit({"error": "ni 'grimblast' ni 'grim' están en el PATH"})

    if code != 0 or not os.path.exists(out_path):
        return _emit({"error": f"la captura falló: {err or code}"})

    return _emit({"ok": True, "image_path": out_path, "message": "📸 Captura de pantalla"})


def cmd_lock(_args: argparse.Namespace) -> int:
    """Lock the session via loginctl (non-blocking); fall back to hyprlock."""
    if shutil.which("loginctl") is not None:
        code, _out, err = _run(["loginctl", "lock-session"])
        if code == 0:
            return _emit({"ok": True, "message": "🔒 Sesión bloqueada."})
        last_err = err or code
    else:
        last_err = "loginctl no está en el PATH"

    if shutil.which("hyprlock") is not None:
        # Detached: do not wait for unlock, or the script would block until then.
        subprocess.Popen(
            ["hyprlock"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return _emit({"ok": True, "message": "🔒 Sesión bloqueada (hyprlock)."})

    return _emit({"error": f"no se pudo bloquear: {last_err}"})


def cmd_status(_args: argparse.Namespace) -> int:
    """Summarize active window, monitors and workspaces."""
    if shutil.which("hyprctl") is None:
        return _emit({"error": "'hyprctl' no está en el PATH"})

    def _json(*hypr_args: str):
        code, out, _err = _run(["hyprctl", "-j", *hypr_args])
        if code != 0 or not out:
            return None
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return None

    lines: list[str] = []

    active = _json("activewindow")
    if active and active.get("class"):
        title = active.get("title", "")
        lines.append(f"🪟 Ventana activa: {active['class']} — {title}".rstrip(" —"))
    else:
        lines.append("🪟 Ventana activa: ninguna")

    monitors = _json("monitors") or []
    for mon in monitors:
        ws = (mon.get("activeWorkspace") or {}).get("id", "?")
        focused = " (enfocado)" if mon.get("focused") else ""
        lines.append(f"🖥️ {mon.get('name', '?')}: workspace {ws}{focused}")

    workspaces = _json("workspaces") or []
    if workspaces:
        lines.append(f"🗂️ Workspaces activos: {len(workspaces)}")

    return _emit({"ok": True, "message": "\n".join(lines)})


def cmd_notify(args: argparse.Namespace) -> int:
    """notify-send <text> on the host desktop."""
    if shutil.which("notify-send") is None:
        return _emit({"error": "'notify-send' no está en el PATH"})

    text = " ".join(args.text).strip()
    if not text:
        return _emit({"error": "Indica el texto del aviso"})

    code, _out, err = _run(["notify-send", "eyetor", text])
    if code != 0:
        return _emit({"error": f"notify-send falló: {err or code}"})

    return _emit({"ok": True, "message": f"🔔 Aviso enviado: {text}"})


def cmd_volume(args: argparse.Namespace) -> int:
    """Control volume via wpctl (PipeWire) or pactl."""
    action = (args.action or "up").lower()

    if shutil.which("wpctl") is not None:
        sink = "@DEFAULT_AUDIO_SINK@"
        if action == "up":
            cmd = ["wpctl", "set-volume", "-l", "1.5", sink, "5%+"]
        elif action == "down":
            cmd = ["wpctl", "set-volume", sink, "5%-"]
        elif action == "mute":
            cmd = ["wpctl", "set-mute", sink, "toggle"]
        elif action.rstrip("%").isdigit():
            cmd = ["wpctl", "set-volume", "-l", "1.5", sink, f"{action.rstrip('%')}%"]
        else:
            return _emit({"error": f"acción de volumen desconocida: {action}"})
        code, _out, err = _run(cmd)
        if code != 0:
            return _emit({"error": f"wpctl falló: {err or code}"})
        code, out, _err = _run(["wpctl", "get-volume", sink])
        return _emit({"ok": True, "message": f"🔊 {out or action}"})

    if shutil.which("pactl") is not None:
        sink = "@DEFAULT_SINK@"
        if action == "up":
            cmd = ["pactl", "set-sink-volume", sink, "+5%"]
        elif action == "down":
            cmd = ["pactl", "set-sink-volume", sink, "-5%"]
        elif action == "mute":
            cmd = ["pactl", "set-sink-mute", sink, "toggle"]
        elif action.rstrip("%").isdigit():
            cmd = ["pactl", "set-sink-volume", sink, f"{action.rstrip('%')}%"]
        else:
            return _emit({"error": f"acción de volumen desconocida: {action}"})
        code, _out, err = _run(cmd)
        if code != 0:
            return _emit({"error": f"pactl falló: {err or code}"})
        return _emit({"ok": True, "message": f"🔊 Volumen: {action}"})

    return _emit({"error": "ni 'wpctl' ni 'pactl' están en el PATH"})


def cmd_media(args: argparse.Namespace) -> int:
    """Control playback via playerctl."""
    if shutil.which("playerctl") is None:
        return _emit({"error": "'playerctl' no está en el PATH"})

    action = (args.action or "play-pause").lower()
    if action not in {"play-pause", "play", "pause", "next", "previous", "prev", "stop"}:
        return _emit({"error": f"acción de medios desconocida: {action}"})
    if action == "prev":
        action = "previous"

    code, _out, err = _run(["playerctl", action])
    if code != 0:
        return _emit({"error": f"playerctl falló: {err or code}"})

    code, status, _err = _run(["playerctl", "status"])
    return _emit({"ok": True, "message": f"🎵 {status or action}"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hypr.py", description="Control Hyprland")
    sub = parser.add_subparsers(dest="command", required=True)

    p_hello = sub.add_parser("hello", help="Wake screens and restore brightness")
    p_hello.set_defaults(func=cmd_hello)

    p_bye = sub.add_parser("bye", help="Turn screens off (DPMS off)")
    p_bye.set_defaults(func=cmd_bye)

    p_shot = sub.add_parser("shot", help="Take a screenshot")
    p_shot.set_defaults(func=cmd_shot)

    p_lock = sub.add_parser("lock", help="Lock the session")
    p_lock.set_defaults(func=cmd_lock)

    p_status = sub.add_parser("status", help="Active window, monitors, workspaces")
    p_status.set_defaults(func=cmd_status)

    p_notify = sub.add_parser("notify", help="Show a desktop notification")
    p_notify.add_argument("text", nargs="*", help="Notification text")
    p_notify.set_defaults(func=cmd_notify)

    p_volume = sub.add_parser("volume", help="Control audio volume")
    p_volume.add_argument("action", nargs="?", help="up|down|mute|<percent>")
    p_volume.set_defaults(func=cmd_volume)

    p_media = sub.add_parser("media", help="Control media playback")
    p_media.add_argument("action", nargs="?", help="play-pause|next|prev|stop")
    p_media.set_defaults(func=cmd_media)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except subprocess.TimeoutExpired as exc:
        return _emit({"error": f"comando agotó el tiempo: {exc.cmd}"})
    except Exception as exc:  # noqa: BLE001 - surface as JSON to the channel
        return _emit({"error": str(exc)})


if __name__ == "__main__":
    sys.exit(main())
