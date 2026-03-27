---
name: filesystem
description: Read, write, list, search, move, copy and delete files and directories on the host filesystem. Use for any file management task.
license: MIT
compatibility: Python 3.11+. Works on Windows, Linux, macOS.
metadata:
  author: eyetor
  version: "1.0"
---

# Filesystem

## When to use this skill
Use when the user asks to:
- Read or edit a file
- Create or delete files/directories
- List the contents of a directory
- Search for files by name or content
- Move or copy files
- Check if a file exists or get its metadata

## Operations and how to use them

### Read a file
```
scripts/fs.py read --path "/path/to/file.txt"
scripts/fs.py read --path "/path/to/file.txt" --lines 1-50   # only lines 1-50
```

### Write / overwrite a file
```
scripts/fs.py write --path "/path/to/file.txt" --content "Hello world"
```

### Append to a file
```
scripts/fs.py append --path "/path/to/file.txt" --content "New line"
```

### List directory contents
```
scripts/fs.py list --path "/path/to/dir"
scripts/fs.py list --path "/path/to/dir" --pattern "*.py"   # glob filter
```

### Search files by name
```
scripts/fs.py find --path "/path/to/dir" --name "*.log"
```

### Search file contents (grep)
```
scripts/fs.py grep --path "/path/to/dir" --pattern "TODO" --ext ".py"
```

### Delete a file or directory
```
scripts/fs.py delete --path "/path/to/file.txt"
scripts/fs.py delete --path "/path/to/dir" --recursive
```

### Move or rename
```
scripts/fs.py move --src "/old/path.txt" --dst "/new/path.txt"
```

### Create directory
```
scripts/fs.py mkdir --path "/path/to/new/dir"
```

### File info (size, modified date, etc.)
```
scripts/fs.py info --path "/path/to/file.txt"
```

## Notes
- Paths can be absolute or relative (relative to CWD)
- Windows paths work: `C:/Users/user/Documents/file.txt`
- For large files, use `--lines` to read a specific range
- `delete --recursive` removes entire directory trees — use with care
- All operations return JSON
