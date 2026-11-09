"""Judging logic of evals/tool_choice.py (no model involved)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

_PATH = Path(__file__).resolve().parents[1] / "evals" / "tool_choice.py"
_spec = importlib.util.spec_from_file_location("tool_choice_eval", _PATH)
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)


def test_expected_tool_and_args():
    case = {"input": "x", "expect": "shopping_list_add", "args": {"text": "Leche"}}
    assert tc.judge(case, [("shopping_list_add", '{"text": "leche entera"}')])[0]
    assert not tc.judge(case, [("shopping_list_add", '{"text": "pan"}')])[0]
    assert not tc.judge(case, [("web_search", '{"query": "leche"}')])[0]
    assert not tc.judge(case, [])[0]


def test_none_and_alternatives():
    assert tc.judge({"input": "x", "expect": "none"}, [])[0]
    assert not tc.judge({"input": "x", "expect": "none"}, [("web_search", "{}")])[0]
    either = {"input": "x", "expect": ["none", "llm__GetDateTime"]}
    assert tc.judge(either, [])[0] and tc.judge(either, [("llm__GetDateTime", "{}")])[0]


def test_legacy_free_text_equivalent_when_typed_tool_not_offered():
    case = {"input": "x", "expect": "shopping_list_show"}
    offered = {"skill_shopping"}
    assert tc.judge(case, [("skill_shopping", '{"args": "list show"}')], offered)[0]
    assert tc.judge(case, [("skill_shopping", '{"args": "list.py show --with-canonical"}')], offered)[0]
    assert not tc.judge(case, [("skill_shopping", '{"args": "list.py list"}')], offered)[0]
    assert not tc.judge(case, [("skill_shopping", "{}")], offered)[0]
    # When the typed tool IS offered, the free-text call no longer counts.
    assert not tc.judge(case, [("skill_shopping", '{"args": "list show"}')], {"shopping_list_show"})[0]


def test_example_cases_are_well_formed():
    cases = yaml.safe_load(tc.EXAMPLE_CASES.read_text(encoding="utf-8"))
    assert cases and all("input" in c and "expect" in c for c in cases)
