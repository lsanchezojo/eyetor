"""Plugin registry — manages plugin lifecycle and hook dispatch."""

from __future__ import annotations

import asyncio
import logging

from eyetor.plugins.hooks import HookDecision, parse_pre_hook_output, run_hook
from eyetor.plugins.loader import discover_plugins
from eyetor.plugins.manifest import PluginManifest

logger = logging.getLogger(__name__)

_LIFECYCLE_TIMEOUT = 30.0


class PluginRegistry:
    """Central registry for plugins.

    Handles discovery, lifecycle (Init/Shutdown), and hook dispatch.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, PluginManifest] = {}

    def load_all(self, plugin_dirs: list[str]) -> None:
        """Discover and register all plugins from the given directories."""
        manifests = discover_plugins(plugin_dirs)
        for m in manifests:
            self._plugins[m.name] = m

    async def run_init(self) -> None:
        """Run lifecycle Init commands for all loaded plugins."""
        for name, plugin in self._plugins.items():
            if plugin.lifecycle_init:
                logger.info("Running init for plugin '%s': %s", name, plugin.lifecycle_init)
                await self._run_lifecycle(plugin, plugin.lifecycle_init)

    async def run_shutdown(self) -> None:
        """Run lifecycle Shutdown commands for all loaded plugins."""
        for name, plugin in self._plugins.items():
            if plugin.lifecycle_shutdown:
                logger.info("Running shutdown for plugin '%s': %s", name, plugin.lifecycle_shutdown)
                await self._run_lifecycle(plugin, plugin.lifecycle_shutdown)

    async def run_pre_hooks(self, tool_name: str, arguments: str) -> HookDecision:
        """Run all pre_tool_use hooks. Returns the combined decision.

        If any hook denies, the tool call is blocked.
        If any hook modifies input, the last modification wins.
        """
        decision = HookDecision(allow=True)
        for plugin in self._plugins.values():
            script = plugin.hooks.get("pre_tool_use")
            if not script:
                continue
            stdout = await run_hook(
                script, event="pre_tool_use",
                tool_name=tool_name, tool_input=arguments,
            )
            result = parse_pre_hook_output(stdout)
            if result.deny:
                logger.info(
                    "Plugin '%s' denied tool '%s': %s",
                    plugin.name, tool_name, result.deny_reason,
                )
                return result  # deny immediately
            if result.modified_input:
                decision.modified_input = result.modified_input
        return decision

    async def run_post_hooks(
        self, tool_name: str, arguments: str, result: str, duration_ms: int | None = None,
    ) -> None:
        """Run all post_tool_use hooks (fire & forget)."""
        for plugin in self._plugins.values():
            script = plugin.hooks.get("post_tool_use")
            if not script:
                continue
            await run_hook(
                script, event="post_tool_use",
                tool_name=tool_name, tool_input=arguments, tool_result=result,
                tool_duration_ms=duration_ms,
            )

    async def run_failure_hooks(
        self, tool_name: str, arguments: str, error: str, duration_ms: int | None = None,
    ) -> None:
        """Run all post_tool_use_failure hooks (fire & forget)."""
        for plugin in self._plugins.values():
            script = plugin.hooks.get("post_tool_use_failure")
            if not script:
                continue
            await run_hook(
                script, event="post_tool_use_failure",
                tool_name=tool_name, tool_input=arguments, tool_error=error,
                tool_duration_ms=duration_ms,
            )

    def list_plugins(self) -> list[PluginManifest]:
        return list(self._plugins.values())

    async def _run_lifecycle(self, plugin: PluginManifest, command: str) -> None:
        """Run a lifecycle command (init/shutdown) in the plugin directory."""
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(plugin.path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_LIFECYCLE_TIMEOUT
            )
            if proc.returncode != 0:
                logger.warning(
                    "Plugin '%s' lifecycle command failed (%d): %s",
                    plugin.name, proc.returncode, stderr.decode(errors="replace")[:500],
                )
        except asyncio.TimeoutError:
            logger.warning(
                "Plugin '%s' lifecycle command timed out after %.0fs",
                plugin.name, _LIFECYCLE_TIMEOUT,
            )
        except Exception as exc:
            logger.warning("Plugin '%s' lifecycle error: %s", plugin.name, exc)
