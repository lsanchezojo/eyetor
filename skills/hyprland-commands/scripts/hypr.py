#!/usr/bin/env python3
"""Control Hyprland from the chat.

Usage:
  hypr.py hello              Wake screens (DPMS on) and restore saved brightness.
  hypr.py bye                Turn screens off (DPMS off).
  hypr.py shot [workspace]   Screenshot a workspace (default: active) -> image_path.
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


def _hypr_json(*hypr_args: str):
    """Run `hyprctl -j <args>` and return the parsed JSON, or None on failure."""
    code, out, _err = _run(["hyprctl", "-j", *hypr_args])
    if code != 0 or not out:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


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


def _grim_output(output: str, out_path: str) -> tuple[int, str]:
    """Capture a single Hyprland output with grim. Returns (code, err)."""
    code, _out, err = _run(["grim", "-o", output, out_path], timeout=20.0)
    return code, err


def _focused_monitor(monitors: list[dict]) -> dict | None:
    for mon in monitors:
        if mon.get("focused"):
            return mon
    return monitors[0] if monitors else None


def cmd_shot(args: argparse.Namespace) -> int:
    """Screenshot a Hyprland workspace (default: active) and return its path."""
    if shutil.which("hyprctl") is None:
        return _emit({"error": "'hyprctl' no está en el PATH"})

    out_path = os.path.join(
        tempfile.gettempdir(), f"eyetor-shot-{time.time_ns()}.png"
    )
    target = args.workspace

    monitors = _hypr_json("monitors") or []

    # --- No parameter: capture the active workspace (focused output) ---
    if target is None:
        if shutil.which("grim") is None:
            # Fallback: whole screen via grimblast when grim isn't available.
            if shutil.which("grimblast") is None:
                return _emit({"error": "ni 'grim' ni 'grimblast' están en el PATH"})
            code, _out, err = _run(["grimblast", "save", "screen", out_path], timeout=20.0)
            if code != 0 or not os.path.exists(out_path):
                return _emit({"error": f"la captura falló: {err or code}"})
            return _emit({"ok": True, "image_path": out_path, "message": "📸 Captura de pantalla"})

        mon = _focused_monitor(monitors)
        if mon is None:
            return _emit({"error": "no se encontró ningún monitor"})
        ws = (mon.get("activeWorkspace") or {}).get("id", "?")
        code, err = _grim_output(mon["name"], out_path)
        if code != 0 or not os.path.exists(out_path):
            return _emit({"error": f"la captura falló: {err or code}"})
        return _emit({
            "ok": True,
            "image_path": out_path,
            "message": f"📸 Captura (workspace {ws} @ {mon['name']})",
        })

    # --- With parameter: capture a specific workspace ---
    if shutil.which("grim") is None:
        return _emit({"error": "'grim' es necesario para capturar un workspace concreto"})

    target_id = int(target) if str(target).lstrip("-").isdigit() else target

    workspaces = _hypr_json("workspaces") or []
    if not any(ws.get("id") == target_id for ws in workspaces):
        return _emit({"error": f"El workspace {target} no existe"})

    def _monitor_showing(ws_id, mons: list[dict]) -> dict | None:
        for mon in mons:
            if (mon.get("activeWorkspace") or {}).get("id") == ws_id:
                return mon
        return None

    # Already visible somewhere: capture directly, no switching.
    visible = _monitor_showing(target_id, monitors)
    if visible is not None:
        code, err = _grim_output(visible["name"], out_path)
        if code != 0 or not os.path.exists(out_path):
            return _emit({"error": f"la captura falló: {err or code}"})
        return _emit({
            "ok": True,
            "image_path": out_path,
            "message": f"📸 Captura (workspace {target_id} @ {visible['name']})",
        })

    # Not visible: switch to it momentarily, capture, then restore.
    focused = _focused_monitor(monitors)
    if focused is None:
        return _emit({"error": "no se encontró ningún monitor"})
    prev_ws = (focused.get("activeWorkspace") or {}).get("id")

    # Disable the switch animation so the capture isn't caught mid-transition
    # (otherwise the frame shows a sliver of the previous/next workspace).
    anim = _hypr_json("getoption", "animations:enabled")
    prev_anim = anim.get("int", 1) if isinstance(anim, dict) else 1

    try:
        _run(["hyprctl", "keyword", "animations:enabled", "0"])
        code, _out, err = _run(["hyprctl", "dispatch", "workspace", str(target_id)])
        if code != 0:
            return _emit({"error": f"no se pudo cambiar al workspace {target_id}: {err or code}"})
        time.sleep(0.12)  # let the new frame render (no animation now)

        now = _hypr_json("monitors") or monitors
        shown = _monitor_showing(target_id, now) or _focused_monitor(now)
        if shown is None:
            return _emit({"error": "no se encontró el monitor tras conmutar"})
        code, err = _grim_output(shown["name"], out_path)
        if code != 0 or not os.path.exists(out_path):
            return _emit({"error": f"la captura falló: {err or code}"})
        return _emit({
            "ok": True,
            "image_path": out_path,
            "message": f"📸 Captura (workspace {target_id} @ {shown['name']})",
        })
    finally:
        # Restore the workspace while animations are still off (no flicker),
        # then re-enable animations.
        if prev_ws is not None:
            _run(["hyprctl", "dispatch", "focusmonitor", focused["name"]])
            _run(["hyprctl", "dispatch", "workspace", str(prev_ws)])
        _run(["hyprctl", "keyword", "animations:enabled", str(prev_anim)])


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

    lines: list[str] = []

    active = _hypr_json("activewindow")
    if active and active.get("class"):
        title = active.get("title", "")
        lines.append(f"🪟 Ventana activa: {active['class']} — {title}".rstrip(" —"))
    else:
        lines.append("🪟 Ventana activa: ninguna")

    monitors = _hypr_json("monitors") or []
    for mon in monitors:
        ws = (mon.get("activeWorkspace") or {}).get("id", "?")
        focused = " (enfocado)" if mon.get("focused") else ""
        lines.append(f"🖥️ {mon.get('name', '?')}: workspace {ws}{focused}")

    workspaces = _hypr_json("workspaces") or []
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

    p_shot = sub.add_parser("shot", help="Screenshot a workspace (default: active)")
    p_shot.add_argument(
        "workspace", nargs="?", help="número de workspace; vacío = el activo"
    )
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
