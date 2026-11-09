---
name: filesystem
description: Read, write, list, search, move and delete files and directories on the host filesystem. Use for any file management task.
license: MIT
compatibility: Python 3.11+. Works on Windows, Linux, macOS.
metadata:
  author: eyetor
  version: "2.0"
# Only typed tools (fs_*): no free-text skill_filesystem tool.
args_tool: false
tools:
  - name: fs_read
    description: Read a text file. Relative paths resolve against ~/.eyetor.
    script: fs.py
    argv: [read]
    parameters:
      type: object
      properties:
        path: {type: string, description: File path.}
        lines: {type: string, description: "Optional line range, e.g. '1-50' or '120'."}
      required: [path]
  - name: fs_write
    description: Create or overwrite a text file with the given content.
    script: fs.py
    argv: [write]
    parameters:
      type: object
      properties:
        path: {type: string, description: File path.}
        content: {type: string, description: Full file content.}
      required: [path, content]
  - name: fs_append
    description: Append text to the end of a file (creates it if missing).
    script: fs.py
    argv: [append]
    parameters:
      type: object
      properties:
        path: {type: string, description: File path.}
        content: {type: string, description: Text to append.}
      required: [path, content]
  - name: fs_list
    description: List a directory's entries with type, size and modification date.
    script: fs.py
    argv: [list]
    parameters:
      type: object
      properties:
        path: {type: string, description: Directory path.}
        pattern: {type: string, description: "Optional glob filter, e.g. '*.py'."}
      required: [path]
  - name: fs_find
    description: Find files by name (recursive glob) under a directory.
    script: fs.py
    argv: [find]
    parameters:
      type: object
      properties:
        path: {type: string, description: Directory to search.}
        name: {type: string, description: "Name glob, e.g. '*.log'."}
      required: [path, name]
  - name: fs_grep
    description: Search file contents (case-insensitive regex) under a directory, returning file, line and text.
    script: fs.py
    argv: [grep]
    parameters:
      type: object
      properties:
        path: {type: string, description: Directory to search.}
        pattern: {type: string, description: Regular expression.}
        ext: {type: string, description: "Optional extension filter, e.g. '.py'."}
      required: [path, pattern]
  - name: fs_delete
    description: Delete a file or directory. Destructive — confirm with the user first.
    script: fs.py
    argv: [delete]
    confirm: true
    parameters:
      type: object
      properties:
        path: {type: string, description: Path to delete.}
        recursive: {type: boolean, description: Required to delete a non-empty directory.}
      required: [path]
  - name: fs_move
    description: Move or rename a file or directory.
    script: fs.py
    argv: [move]
    parameters:
      type: object
      properties:
        src: {type: string, description: Source path.}
        dst: {type: string, description: Destination path.}
      required: [src, dst]
  - name: fs_mkdir
    description: Create a directory (and missing parents).
    script: fs.py
    argv: [mkdir]
    parameters:
      type: object
      properties:
        path: {type: string, description: Directory path.}
      required: [path]
  - name: fs_info
    description: Get metadata for a path (type, size, created/modified dates, extension).
    script: fs.py
    argv: [info]
    parameters:
      type: object
      properties:
        path: {type: string, description: Path.}
      required: [path]
---

# Filesystem

File management through the typed `fs_*` tools, all backed by `scripts/fs.py`.

## Tools

| Tool | What it does |
|------|--------------|
| `fs_read` | Read a text file, optionally a line range (`lines: "1-50"`) |
| `fs_write` / `fs_append` | Overwrite / append text |
| `fs_list` | List a directory, optional glob filter |
| `fs_find` | Find files by name glob, recursively |
| `fs_grep` | Search contents with a regex, optional extension filter |
| `fs_delete` | Delete a file, or a directory with `recursive: true` (asks for confirmation) |
| `fs_move` | Move or rename |
| `fs_mkdir` | Create a directory and its parents |
| `fs_info` | Size, dates and type of a path |

## Notes
- Relative paths (and `.`) resolve against eyetor's data directory `~/.eyetor`, not the skill's own folder. Use an absolute path for any other location.
- Binary files (images, PDFs…) are refused by `fs_read`: images are already described by the vision model.
- All operations return JSON (`{"ok": true, ...}` or `{"ok": false, "error": ...}`).
- The script can still be run by hand: `scripts/fs.py read --path FILE [--lines A-B]`, etc. (see its docstring).
