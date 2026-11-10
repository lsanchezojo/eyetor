"""Detects a model stuck calling the same tool with (nearly) the same args."""

from __future__ import annotations

import json
import re

from eyetor.models.messages import ToolCall


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Jaccard threshold for the "soft-loop" guard: three consecutive same-tool
# calls whose token-bags pairwise overlap above this ratio are treated as a
# loop, catching cases the exact-match check misses (same core, varying
# satellite words — see _tool_call_bag docstring).
_LOOP_JACCARD_THRESHOLD = 0.6


def _tool_call_bag(name: str, arguments: str) -> tuple[str, frozenset[str]]:
    """Return (tool_name, lowercase token-bag) for a tool call.

    Used by both the exact-match signature (below) and the Jaccard soft-loop
    check. Small models often permute the same keywords into dozens of
    "different" queries; this reduces them to a comparable bag.
    """
    try:
        args = json.loads(arguments) if arguments else {}
    except (ValueError, TypeError):
        args = arguments

    def walk(obj) -> list[str]:
        if isinstance(obj, str):
            return [t.lower() for t in _TOKEN_RE.findall(obj)]
        if isinstance(obj, dict):
            out: list[str] = []
            for v in obj.values():
                out.extend(walk(v))
            return out
        if isinstance(obj, list):
            out = []
            for v in obj:
                out.extend(walk(v))
            return out
        return []

    return name, frozenset(walk(args))


def _normalize_tool_call(name: str, arguments: str) -> str:
    """Collapse a tool call to a sorted-token signature for exact-match loop
    detection. Captures pure permutations (``a b c`` ≡ ``c a b``).
    """
    _, tokens = _tool_call_bag(name, arguments)
    return f"{name}:{','.join(sorted(tokens))}"


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


class _LoopDetector:
    """Detects a turn stuck calling the same tools. Two layers:

    1. Exact match on normalized token-bag signature (catches pure keyword
       permutations: "a b c" ≡ "c a b").
    2. Jaccard soft-loop: N same-tool calls whose token bags pairwise overlap
       ≥ threshold (catches "same core + varying satellite words" — the real
       failure mode of small models).
    """

    def __init__(self, max_repeat: int = 3) -> None:
        self.max_repeat = max_repeat  # consecutive similar calls before forcing an answer
        self._calls: list[str] = []  # normalized sig for exact-match detection
        self._bags: list[list[tuple[str, frozenset[str]]]] = []  # for Jaccard

    def observe(self, tool_calls: list[ToolCall]) -> str | None:
        """Record this iteration's calls; return a reason string if looping."""
        signatures = [
            _normalize_tool_call(tc.function.name, tc.function.arguments) for tc in tool_calls
        ]
        self._calls.append("|".join(sorted(signatures)))
        self._bags.append(
            [_tool_call_bag(tc.function.name, tc.function.arguments) for tc in tool_calls]
        )
        self._calls = self._calls[-self.max_repeat:]
        self._bags = self._bags[-self.max_repeat:]

        if len(self._calls) == self.max_repeat and len(set(self._calls)) == 1:
            return f"exact repetition of {[tc.function.name for tc in tool_calls]}"
        if (
            len(self._bags) == self.max_repeat
            and all(len(b) == 1 for b in self._bags)
            and len({b[0][0] for b in self._bags}) == 1
        ):
            # Soft loop: among the last N same-tool calls, require every
            # pair to overlap above threshold. This still catches true
            # repeated loops while allowing legitimate retries that change
            # strategy, parameters, or timeout after a failed attempt.
            bags = [b[0][1] for b in self._bags]
            pairwise = [
                _jaccard(bags[i], bags[j])
                for i in range(len(bags))
                for j in range(i + 1, len(bags))
            ]
            min_jaccard = min(pairwise) if pairwise else 0.0
            if pairwise and min_jaccard >= _LOOP_JACCARD_THRESHOLD:
                return (
                    f"{self.max_repeat} near-duplicate '{self._bags[-1][0][0]}' "
                    f"calls (min Jaccard {min_jaccard:.2f} ≥ {_LOOP_JACCARD_THRESHOLD})"
                )
        return None
