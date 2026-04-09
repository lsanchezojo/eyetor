"""ConversationCompactor — manages context window via summarization."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from eyetor.models.messages import Message

if TYPE_CHECKING:
    from eyetor.config import CompactionConfig
    from eyetor.providers.base import BaseProvider

logger = logging.getLogger(__name__)

PROMPT_SUMMARY = """Summarize the following conversation history.

## Goal
What the user is trying to accomplish.

## Key Discoveries
Important findings, file paths, errors, resolutions.

## Work Completed
Actions taken and decisions made.

## Current State
What is done, what is pending.

## Next Steps (if known)

IMPORTANT: Preserve verbatim:
- Exact file paths and commands executed
- Error messages and their resolutions
- API endpoints, URLs, and credentials used
- Exact values (IDs, versions, timestamps)

Omit pleasantries and filler text. Be specific."""


@dataclass
class CompactionResult:
    """Result of a compaction operation."""

    compacted: bool
    new_messages: list[Message]
    summary_text: str | None = None
    phase: int | None = None
    archived_path: Path | None = None


class ConversationCompactor:
    """Two-phase conversation compactor with verbatim tail."""

    def __init__(self, config: CompactionConfig) -> None:
        self._config = config

    def estimate_tokens(self, messages: list[Message], system_content: str) -> int:
        """Estimate token count for messages + system prompt.

        Uses tiktoken if available, otherwise falls back to character count
        with adaptive ratio based on context window size.
        """
        total_chars = len(system_content)
        for msg in messages:
            total_chars += len(msg.content or "")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total_chars += len(tc.function.name) + len(tc.function.arguments)

        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return enc.encode(system_content).__len__() + sum(
                enc.encode(msg.content or "").__len__() for msg in messages
            )
        except ImportError:
            ratio = 3.0 if self._config.context_window > 100_000 else 4.0
            return int(total_chars / ratio)

    def should_compact(self, messages: list[Message], system_content: str) -> bool:
        """Return True if context exceeds trigger threshold."""
        if not self._config.enabled:
            logger.debug("Compaction disabled")
            return False
        tokens = self.estimate_tokens(messages, system_content)
        threshold = self._config.context_window * self._config.trigger_at_percent
        result = tokens >= threshold
        logger.info(
            "Compaction check: %d tokens, threshold %d, compact=%s",
            tokens,
            int(threshold),
            result,
        )
        return result

    async def compact(
        self,
        messages: list[Message],
        system_content: str,
        provider: BaseProvider,
        session_id: str,
    ) -> CompactionResult:
        """Execute two-phase compaction."""
        threshold = self._config.context_window * self._config.trigger_at_percent

        history, tail = self._split_tail(messages)
        if not history:
            return CompactionResult(compacted=False, new_messages=messages)

        pruned = self._prune_tool_outputs(history)
        tokens = self.estimate_tokens(pruned + tail, system_content)

        if tokens >= threshold:
            try:
                summary = await self._summarize(history, provider)
            except Exception as e:
                logger.warning(
                    "Compaction LLM failed: %s, using aggressive truncate",
                    e,
                )
                pruned = [m for m in pruned if len(m.content or "") < 500]
                tokens = self.estimate_tokens(pruned + tail, system_content)

                if tokens >= threshold:
                    logger.warning(
                        "Compaction insufficient after fallback, using tail only"
                    )
                    return CompactionResult(
                        compacted=True,
                        new_messages=tail,
                        phase=2,
                    )

            max_summary_chars = int(
                self._config.context_window * self._config.summary_max_percent * 4
            )

            if len(summary) > max_summary_chars:
                summary = summary[:max_summary_chars] + "\n[truncated]"

            summary_msg = Message(role="system", content=summary)
            new_messages = [summary_msg] + tail

            final_tokens = self.estimate_tokens(new_messages, system_content)
            if final_tokens >= threshold:
                logger.warning(
                    "Compaction still exceeds threshold after summarization, using tail only"
                )
                archive_path = self._archive(messages, session_id)
                return CompactionResult(
                    compacted=True,
                    new_messages=tail,
                    phase=2,
                    archived_path=archive_path,
                )

            archive_path = self._archive(messages, session_id)

            return CompactionResult(
                compacted=True,
                new_messages=new_messages,
                summary_text=summary,
                phase=2,
                archived_path=archive_path,
            )

        if pruned != history:
            new_messages = pruned + tail
            archive_path = self._archive(messages, session_id)
            return CompactionResult(
                compacted=True,
                new_messages=new_messages,
                phase=1,
                archived_path=archive_path,
            )

        return CompactionResult(compacted=False, new_messages=messages)

    def _split_tail(
        self, messages: list[Message]
    ) -> tuple[list[Message], list[Message]]:
        """Split messages into history and verbatim tail by user turns."""
        user_positions = [i for i, m in enumerate(messages) if m.role == "user"]
        if len(user_positions) <= self._config.keep_last_n_user_turns:
            return [], messages

        split_idx = user_positions[-self._config.keep_last_n_user_turns]
        return messages[:split_idx], messages[split_idx:]

    def _prune_tool_outputs(self, messages: list[Message]) -> list[Message]:
        """Truncate tool outputs exceeding max_chars."""
        pruned = []
        for msg in messages:
            if msg.role == "tool" and msg.content:
                if len(msg.content) > self._config.tool_output_max_chars:
                    pruned.append(
                        Message(
                            role="tool",
                            content=msg.content[: self._config.tool_output_max_chars]
                            + f"\n[truncated from {len(msg.content)} chars]",
                            tool_call_id=msg.tool_call_id,
                        )
                    )
                else:
                    pruned.append(msg)
            else:
                pruned.append(msg)
        return pruned

    async def _summarize(self, history: list[Message], provider: BaseProvider) -> str:
        """Call LLM to summarize conversation history."""
        raw_provider = getattr(provider, "_inner", provider)

        prompt = f"{PROMPT_SUMMARY}\n\n---\n\n{self._serialize_for_summary(history)}"

        result = await raw_provider.complete(
            messages=[Message(role="user", content=prompt)],
            tools=None,
            temperature=0.0,
        )

        return result.message.content or ""

    def _serialize_for_summary(self, messages: list[Message]) -> str:
        """Serialize messages to plain text for summarization prompt."""
        lines = []
        for msg in messages:
            role = msg.role
            content = msg.content or ""

            if msg.role == "tool":
                lines.append(f"[Tool result]: {content}")
            elif msg.tool_calls:
                calls = ", ".join(
                    f"{tc.function.name}({tc.function.arguments})"
                    for tc in msg.tool_calls
                )
                lines.append(f"[Assistant called]: {calls}")
            else:
                lines.append(f"[{role.upper()}]: {content}")

        return "\n".join(lines)

    def _archive(self, messages: list[Message], session_id: str) -> Path | None:
        """Write pre-compaction messages to archive file."""
        if not self._config.archive_dir:
            logger.debug("Archive disabled (archive_dir not set)")
            return None

        archive_dir = Path(self._config.archive_dir).expanduser()
        logger.debug("Creating archive in: %s", archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)

        safe_id = re.sub(r"[/:@\\]", "_", session_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{safe_id}.{timestamp}.pre-compaction.jsonl"
        archive_path = archive_dir / archive_name

        with open(archive_path, "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(
                    json.dumps(msg.model_dump(exclude_none=True), ensure_ascii=False)
                    + "\n"
                )

        logger.info("Archived pre-compaction messages to %s", archive_path)
        return archive_path
