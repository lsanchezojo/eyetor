#!/usr/bin/env python3
"""Filesystem operations for the filesystem skill.

Usage:
    fs.py read   --path FILE [--lines START-END]
    fs.py write  --path FILE --content TEXT
    fs.py append --path FILE --content TEXT
    fs.py list   --path DIR  [--pattern GLOB]
    fs.py find   --path DIR  --name PATTERN
    fs.py grep   --path DIR  --pattern REGEX [--ext EXT]
    fs.py delete --path PATH [--recursive]
    fs.py move   --src SRC   --dst DST
    fs.py mkdir  --path DIR
    fs.py info   --path PATH
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


def _ok(data) -> None:
    print(json.dumps({"ok": True, **data} if isinstance(data, dict) else {"ok": True, "result": data}, ensure_ascii=False))


def _err(msg: str) -> None:
    print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))
    sys.exit(1)


def cmd_read(path: str, lines: str | None) -> None:
    p = Path(path)
    if not p.exists():
        _err(f"File not found: {path}")
    if not p.is_file():
        _err(f"Not a file: {path}")
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        _err(str(e))
    if lines:
        # Parse "start-end" or "start"
        parts = lines.split("-")
        start = int(parts[0]) - 1
        end = int(parts[1]) if len(parts) > 1 else start + 1
        all_lines = text.splitlines()
        text = "\n".join(all_lines[start:end])
    _ok({"path": str(p.resolve()), "content": text, "size_bytes": p.stat().st_size})


def cmd_write(path: str, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(content, encoding="utf-8")
        _ok({"path": str(p.resolve()), "bytes_written": len(content.encode())})
    except Exception as e:
        _err(str(e))


def cmd_append(path: str, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with p.open("a", encoding="utf-8") as f:
            f.write(content)
        _ok({"path": str(p.resolve()), "bytes_appended": len(content.encode())})
    except Exception as e:
        _err(str(e))


def cmd_list(path: str, pattern: str | None) -> None:
    p = Path(path)
    if not p.exists():
        _err(f"Path not found: {path}")
    entries = []
    for item in sorted(p.iterdir()):
        if pattern and not fnmatch.fnmatch(item.name, pattern):
            continue
        stat = item.stat()
        entries.append({
            "name": item.name,
            "type": "dir" if item.is_dir() else "file",
            "size_bytes": stat.st_size if item.is_file() else None,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    _ok({"path": str(p.resolve()), "entries": entries, "count": len(entries)})


def cmd_find(path: str, name: str) -> None:
    p = Path(path)
    if not p.exists():
        _err(f"Path not found: {path}")
    matches = [str(f) for f in p.rglob(name)]
    _ok({"matches": matches, "count": len(matches)})


def cmd_grep(path: str, pattern: str, ext: str | None) -> None:
    p = Path(path)
    if not p.exists():
        _err(f"Path not found: {path}")
    results = []
    glob = f"**/*{ext}" if ext else "**/*"
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        _err(f"Invalid regex: {e}")
    for f in p.rglob("*"):
        if not f.is_file():
            continue
        if ext and f.suffix != ext:
            continue
        try:
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    results.append({"file": str(f), "line": i, "content": line.strip()})
        except Exception:
            pass
    _ok({"matches": results, "count": len(results)})


def cmd_delete(path: str, recursive: bool) -> None:
    p = Path(path)
    if not p.exists():
        _err(f"Path not found: {path}")
    try:
        if p.is_dir():
            if recursive:
                shutil.rmtree(p)
            else:
                p.rmdir()
        else:
            p.unlink()
        _ok({"deleted": str(p.resolve())})
    except Exception as e:
        _err(str(e))


def cmd_move(src: str, dst: str) -> None:
    s, d = Path(src), Path(dst)
    if not s.exists():
        _err(f"Source not found: {src}")
    try:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
        _ok({"moved": {"from": str(s.resolve()), "to": str(d.resolve())}})
    except Exception as e:
        _err(str(e))


def cmd_mkdir(path: str) -> None:
    p = Path(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
        _ok({"created": str(p.resolve())})
    except Exception as e:
        _err(str(e))


def cmd_info(path: str) -> None:
    p = Path(path)
    if not p.exists():
        _err(f"Path not found: {path}")
    stat = p.stat()
    _ok({
        "path": str(p.resolve()),
        "type": "dir" if p.is_dir() else "file",
        "size_bytes": stat.st_size,
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "extension": p.suffix,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    # read
    p = sub.add_parser("read"); p.add_argument("--path", required=True); p.add_argument("--lines")
    # write
    p = sub.add_parser("write"); p.add_argument("--path", required=True); p.add_argument("--content", required=True)
    # append
    p = sub.add_parser("append"); p.add_argument("--path", required=True); p.add_argument("--content", required=True)
    # list
    p = sub.add_parser("list"); p.add_argument("--path", required=True); p.add_argument("--pattern")
    # find
    p = sub.add_parser("find"); p.add_argument("--path", required=True); p.add_argument("--name", required=True)
    # grep
    p = sub.add_parser("grep"); p.add_argument("--path", required=True); p.add_argument("--pattern", required=True); p.add_argument("--ext")
    # delete
    p = sub.add_parser("delete"); p.add_argument("--path", required=True); p.add_argument("--recursive", action="store_true")
    # move
    p = sub.add_parser("move"); p.add_argument("--src", required=True); p.add_argument("--dst", required=True)
    # mkdir
    p = sub.add_parser("mkdir"); p.add_argument("--path", required=True)
    # info
    p = sub.add_parser("info"); p.add_argument("--path", required=True)

    args = parser.parse_args()

    if args.command == "read":       cmd_read(args.path, getattr(args, "lines", None))
    elif args.command == "write":    cmd_write(args.path, args.content)
    elif args.command == "append":   cmd_append(args.path, args.content)
    elif args.command == "list":     cmd_list(args.path, getattr(args, "pattern", None))
    elif args.command == "find":     cmd_find(args.path, args.name)
    elif args.command == "grep":     cmd_grep(args.path, args.pattern, getattr(args, "ext", None))
    elif args.command == "delete":   cmd_delete(args.path, getattr(args, "recursive", False))
    elif args.command == "move":     cmd_move(args.src, args.dst)
    elif args.command == "mkdir":    cmd_mkdir(args.path)
    elif args.command == "info":     cmd_info(args.path)


if __name__ == "__main__":
    main()
