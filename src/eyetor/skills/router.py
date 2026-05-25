"""Automatic script routing for skills.

Given raw CLI args, determines which script to execute and splits out
the remaining arguments.  Single-script skills pass everything through;
multi-script skills use the first token to select the target script.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

# Shell variable references: $VAR, "$VAR", ${VAR}, "${VAR}"
_SHELL_VAR_RE = re.compile(r'^"?\$\{?[A-Za-z_][A-Za-z0-9_]*\}?"?$')
# Absolute or home-relative paths that are script invocations, not commands
_ABS_PATH_RE = re.compile(r'^[/~]')
_PYTHON_RE = re.compile(r"^python(?:\d+(?:\.\d+)*)?(?:\.exe)?$")


class RoutingError(Exception):
    """Raised when the router cannot determine which script to run."""


class ScriptRouter:
    """Routes raw args to the correct script inside a skill."""

    def __init__(self, skill_name: str, scripts: list[Path]) -> None:
        self._skill_name = skill_name
        self._scripts = [s for s in scripts if s.is_file() and not s.name.startswith(("_", "."))]
        self._by_name: dict[str, Path] = {s.name: s for s in self._scripts}
        self._by_stem: dict[str, Path] = {s.stem: s for s in self._scripts}

    @property
    def public_scripts(self) -> list[Path]:
        return list(self._scripts)

    def route(self, raw_args: str) -> tuple[Path, list[str]]:
        """Resolve *raw_args* to ``(script_path, arg_list)``.

        Raises :class:`RoutingError` when the target script cannot be
        determined (multi-script skill with ambiguous first token).
        """
        tokens = self._tokenize(raw_args)
        tokens = self._strip_invocation_prefix(tokens)

        # Single-script skill: pass args through after removing accidental
        # script invocations copied from docs, e.g. ``run.py --cmd ...``.
        if len(self._scripts) == 1:
            script = self._scripts[0]
            return script, self._strip_script_invocation(tokens, script)

        # Multi-script: need at least one token to pick a script.
        if not tokens:
            names = ", ".join(sorted(self._by_stem))
            raise RoutingError(
                f"Skill '{self._skill_name}' has multiple scripts. "
                f"First word of args must be one of: {names}. "
                f"Example: \"{next(iter(self._by_stem))} --help\""
            )

        candidate = tokens[0]

        # Strip common prefixes from SKILL.md examples (e.g. "scripts/gmail.py")
        if candidate.startswith("scripts/"):
            candidate = candidate[len("scripts/"):]

        # Try exact filename match ("gmail.py")
        if candidate in self._by_name:
            return self._by_name[candidate], tokens[1:]

        # Try stem match ("gmail")
        if candidate in self._by_stem:
            return self._by_stem[candidate], tokens[1:]

        names = ", ".join(sorted(self._by_stem))
        raise RoutingError(
            f"Unknown script '{tokens[0]}' in skill '{self._skill_name}'. "
            f"First word of args must be one of: {names}."
        )

    @staticmethod
    def _strip_invocation_prefix(tokens: list[str]) -> list[str]:
        """Strip leading tokens that are shell invocation artifacts.

        SKILL.md examples often show: "$PWCLI" open https://...
        The skill tool already handles running the script; only subcommands
        and flags should be passed as args.  Strips:
        - Shell variable references: $VAR, "$VAR", ${VAR}
        - Absolute/home-relative paths: /path/to/script.sh, ~/...
        """
        while tokens and (
            _SHELL_VAR_RE.match(tokens[0]) or _ABS_PATH_RE.match(tokens[0])
        ):
            tokens = tokens[1:]
        return tokens

    @staticmethod
    def _strip_script_invocation(tokens: list[str], script: Path) -> list[str]:
        """Strip a leading invocation of *script* from already-routed args."""
        if not tokens:
            return tokens

        if _is_script_token(tokens[0], script):
            return tokens[1:]

        if _is_interpreter_token(tokens[0]):
            for idx, token in enumerate(tokens[1:], start=1):
                if _is_script_token(token, script):
                    return tokens[idx + 1:]
                if not token.startswith("-"):
                    break

        return tokens

    @staticmethod
    def _tokenize(raw: str) -> list[str]:
        raw = raw.strip()
        if not raw:
            return []
        try:
            return shlex.split(raw)
        except ValueError:
            return raw.split()


def _is_script_token(token: str, script: Path) -> bool:
    """Return True for tokens that explicitly name the routed script file."""
    token_path = token.removeprefix("scripts/")
    return Path(token_path).name == script.name


def _is_interpreter_token(token: str) -> bool:
    """Return True for common interpreter wrappers before script names."""
    name = Path(token).name.lower()
    return bool(_PYTHON_RE.match(name)) or name in {"py", "py.exe"}
