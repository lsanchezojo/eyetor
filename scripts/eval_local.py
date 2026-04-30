"""Tiny local evaluation catalog for Eyetor.

Runs without real LLM servers; this is a prompt/task manifest smoke tool for
tracking representative local-model scenarios.
"""

from __future__ import annotations

import json

EVALS = [
    {"name": "chat", "prompt": "hola, responde breve", "expected_route": "chat", "expected_tools": []},
    {"name": "kb", "prompt": "busca en la KB el procedimiento de prueba", "expected_route": "kb_query", "expected_tools": ["kb_search"]},
    {"name": "empty_web", "prompt": "busca en web algo inexistente", "expected_route": "tool_task", "expected_tools": ["skill_web_search"]},
    {"name": "shell", "prompt": "dime la hora con un comando", "expected_route": "tool_task", "expected_tools": ["skill_shell"]},
    {"name": "scheduler", "prompt": "recuerdame manana a las 9 comprar pan", "expected_route": "tool_task", "expected_tools": ["schedule_task"]},
    {"name": "textual_tool_call", "prompt": "<tool_call>{\"name\":\"kb_search\",\"arguments\":{\"query\":\"x\"}}</tool_call>", "expected_route": "tool_task", "expected_tools": []},
    {"name": "bad_classifier_json", "prompt": "texto con JSON invalido del classifier", "expected_route": "chat", "expected_tools": []},
    {"name": "long_context", "prompt": "contexto largo con compaction", "expected_route": "tool_task", "expected_tools": []},
    {"name": "vision_mock", "prompt": "describe esta imagen mock", "expected_route": "tool_task", "expected_tools": []},
]


def main() -> None:
    report = [{"name": item["name"], "status": "catalogued"} for item in EVALS]
    print(json.dumps({"count": len(EVALS), "results": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
