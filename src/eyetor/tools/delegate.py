"""``delegate`` tool: route self-contained subtasks to subagents (agents_dirs)."""

from __future__ import annotations

import json
import logging

from eyetor.models.tools import ToolDefinition, ToolRegistry


def register_delegate_tool(tool_registry: ToolRegistry, cfg, agent_reg, prov) -> str:
    """Register ``delegate`` if auto_delegate is on; return its system-prompt section."""
    orch_cfg = cfg.orchestrator
    if not orch_cfg.auto_delegate:
        return ""
    prompt = ""
    _requested = list(orch_cfg.workers)
    _resolved_workers = []
    _logger = logging.getLogger(__name__)
    for _wn in _requested:
        if agent_reg.has(_wn):
            _resolved_workers.append(agent_reg.get(_wn))
        else:
            _logger.warning(
                "orchestrator.workers references unknown agent %r — "
                "skipping (available: %s)",
                _wn, ", ".join(agent_reg.list_names()) or "<none>",
            )

    if not _requested:
        _logger.warning(
            "auto_delegate is enabled but orchestrator.workers is empty — "
            "no delegate tool registered"
        )
    elif not _resolved_workers:
        _logger.warning(
            "auto_delegate is enabled but none of the configured workers "
            "were found in agents_dirs — no delegate tool registered"
        )
    else:
        from eyetor.agents.base import BaseAgent
        from eyetor.models.agents import AgentConfig as _WorkerAgentCfg

        _workers_by_name = {w.name: w for w in _resolved_workers}
        _worker_names_list = list(_workers_by_name.keys())

        async def delegate_handler(worker: str, task: str) -> str:
            wdef = _workers_by_name.get(worker)
            if wdef is None:
                return json.dumps({
                    "error": (
                        f"Unknown worker {worker!r}. "
                        f"Available: {_worker_names_list}"
                    )
                }, ensure_ascii=False)
            if not task or not task.strip():
                return json.dumps(
                    {"error": "task is required"}, ensure_ascii=False
                )
            sub_temp = (
                wdef.temperature
                if wdef.temperature is not None
                else prov.temperature
            )
            sub_model = wdef.model or prov.model
            sub_agent = BaseAgent(
                config=_WorkerAgentCfg(
                    name=wdef.name,
                    provider="",
                    model=sub_model,
                    system_prompt=wdef.system_prompt,
                    temperature=sub_temp,
                ),
                provider=prov,
            )
            try:
                result = await sub_agent.run(task)
            except Exception as exc:  # noqa: BLE001
                return json.dumps(
                    {"error": f"Worker {worker!r} failed: {exc}"},
                    ensure_ascii=False,
                )
            return json.dumps(
                {"worker": wdef.name, "result": result.final_output},
                ensure_ascii=False,
            )

        _workers_desc_lines = [
            f"- {w.name}: {w.description}" for w in _resolved_workers
        ]
        _workers_desc = "\n".join(_workers_desc_lines)

        tool_registry.register(
            ToolDefinition(
                name="delegate",
                description=(
                    "Delegate a self-contained subtask to a specialised "
                    "subagent and return its answer. Use this when the "
                    "subtask clearly matches one of the available workers. "
                    "Do NOT delegate trivial questions you can answer directly.\n\n"
                    f"Available workers:\n{_workers_desc}"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "worker": {
                            "type": "string",
                            "enum": _worker_names_list,
                            "description": (
                                "Name of the subagent to delegate to. "
                                f"One of: {_worker_names_list}"
                            ),
                        },
                        "task": {
                            "type": "string",
                            "description": (
                                "The complete, self-contained subtask "
                                "for the worker. Include all the context "
                                "the worker needs — it cannot see the "
                                "main conversation."
                            ),
                        },
                    },
                    "required": ["worker", "task"],
                },
                handler=delegate_handler,
            )
        )

        prompt = (
            "\n\n## Subagent delegation\n\n"
            "You have a `delegate` tool that routes a subtask to a "
            "specialised subagent. Available subagents:\n\n"
            f"{_workers_desc}\n\n"
            "When the user's request maps to one of these specialities, "
            "call `delegate` with a self-contained `task` — the subagent "
            "cannot see the main conversation, so include the necessary "
            "context. Summarise the worker's answer in your final reply; "
            "do not paste raw JSON. Only delegate when the speciality "
            "clearly fits — answer directly otherwise."
        )

        _logger.info(
            "Delegate tool registered with workers: %s",
            _worker_names_list,
        )
    return prompt
