"""Tests for eyetor.skills.router.ScriptRouter."""

from __future__ import annotations

from pathlib import Path

import pytest

from eyetor.skills.router import RoutingError, ScriptRouter


@pytest.fixture
def tmp_scripts(tmp_path: Path) -> dict[str, Path]:
    """Create fake script files and return a name→path mapping."""
    files = {}
    for name in ("gmail.py", "gcalendar.py", "tasks.py", "_auth.py", ".hidden.sh"):
        p = tmp_path / name
        p.touch()
        files[name] = p
    return files


# ── Single-script skills ──────────────────────────────────────────────


class TestSingleScript:
    def _router(self, tmp_path: Path) -> ScriptRouter:
        script = tmp_path / "browser.py"
        script.touch()
        return ScriptRouter("browser", [script])

    def test_args_pass_through(self, tmp_path: Path) -> None:
        r = self._router(tmp_path)
        path, args = r.route('source --url "https://example.com"')
        assert path.name == "browser.py"
        assert args == ["source", "--url", "https://example.com"]

    def test_empty_args(self, tmp_path: Path) -> None:
        r = self._router(tmp_path)
        path, args = r.route("")
        assert path.name == "browser.py"
        assert args == []

    def test_first_token_matches_stem_still_passes_through(self, tmp_path: Path) -> None:
        """Even if the first token is 'browser', it should NOT be stripped."""
        r = self._router(tmp_path)
        path, args = r.route("browser --help")
        assert path.name == "browser.py"
        assert args == ["browser", "--help"]


# ── Multi-script skills ───────────────────────────────────────────────


class TestMultiScript:
    def _router(self, tmp_scripts: dict[str, Path]) -> ScriptRouter:
        return ScriptRouter("google-workspace", list(tmp_scripts.values()))

    def test_match_by_stem(self, tmp_scripts: dict[str, Path]) -> None:
        r = self._router(tmp_scripts)
        path, args = r.route("gmail list --max 20")
        assert path.name == "gmail.py"
        assert args == ["list", "--max", "20"]

    def test_match_by_filename(self, tmp_scripts: dict[str, Path]) -> None:
        r = self._router(tmp_scripts)
        path, args = r.route("gcalendar.py list --days 7")
        assert path.name == "gcalendar.py"
        assert args == ["list", "--days", "7"]

    def test_match_with_scripts_prefix(self, tmp_scripts: dict[str, Path]) -> None:
        r = self._router(tmp_scripts)
        path, args = r.route("scripts/tasks.py list")
        assert path.name == "tasks.py"
        assert args == ["list"]

    def test_unknown_token_raises(self, tmp_scripts: dict[str, Path]) -> None:
        r = self._router(tmp_scripts)
        with pytest.raises(RoutingError, match="Unknown script"):
            r.route("unknown foo bar")

    def test_empty_args_raises(self, tmp_scripts: dict[str, Path]) -> None:
        r = self._router(tmp_scripts)
        with pytest.raises(RoutingError, match="multiple scripts"):
            r.route("")

    def test_private_scripts_excluded(self, tmp_scripts: dict[str, Path]) -> None:
        r = self._router(tmp_scripts)
        public_names = {s.name for s in r.public_scripts}
        assert "_auth.py" not in public_names
        assert ".hidden.sh" not in public_names

    def test_private_script_not_routable(self, tmp_scripts: dict[str, Path]) -> None:
        r = self._router(tmp_scripts)
        with pytest.raises(RoutingError):
            r.route("_auth check")


# ── Edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_shlex_fallback_on_bad_quoting(self, tmp_path: Path) -> None:
        script = tmp_path / "run.py"
        script.touch()
        r = ScriptRouter("shell", [script])
        # Unmatched quote: should fall back to str.split()
        path, args = r.route('--cmd "unclosed')
        assert path.name == "run.py"
        assert args == ["--cmd", '"unclosed']

    def test_no_public_scripts(self, tmp_path: Path) -> None:
        private = tmp_path / "_internal.py"
        private.touch()
        r = ScriptRouter("empty", [private])
        # No public scripts at all — single-script path won't apply,
        # but len(self._scripts)==0, so multi-script path with empty args
        with pytest.raises((RoutingError, IndexError)):
            r.route("anything")
