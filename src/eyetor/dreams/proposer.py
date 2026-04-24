"""Proposal generator — creates human-readable dream proposals."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from eyetor.dreams.store import DreamAnalysis, DreamProposal, DreamsStore, ProposalStatus

logger = logging.getLogger(__name__)


class ProposalGenerator:
    """Generates formatted proposals from dream analysis."""

    def __init__(self, store: DreamsStore, output_dir: Path) -> None:
        self._store = store
        self._output_dir = Path(output_dir).expanduser()

    def generate_and_save(self, analysis: DreamAnalysis) -> list[int]:
        """Generate proposals from analysis and save to store and file."""
        self._output_dir.mkdir(parents=True, exist_ok=True)

        proposal_ids = []

        for i, proposal in enumerate(analysis.proposals):
            proposal_id = self._store.save_proposal(analysis.date, i + 1, proposal)
            proposal_ids.append(proposal_id)
            logger.info("Saved proposal #%d (id=%d)", i + 1, proposal_id)

        self._write_markdown(analysis, proposal_ids)

        return proposal_ids

    def _write_markdown(self, analysis: DreamAnalysis, proposal_ids: list[int]) -> None:
        """Write proposals to markdown file."""
        filename = self._output_dir / f"{analysis.date}_proposals.md"

        priority_emoji = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
        }

        lines = [
            f"## Sueño - {analysis.date}",
            "",
            "### Actividad del Día",
            f"- Sesiones: {analysis.sessions_count}",
            f"- Tool calls: {analysis.tool_calls_count}",
            f"- Errores: {analysis.errors_count}",
            f"- Costo estimado: ${analysis.total_cost:.4f}",
            "",
            "### Hallazgos",
            "",
        ]

        for i, finding in enumerate(analysis.findings):
            emoji = priority_emoji.get(finding.priority.value, "⚪")
            lines.append(f"{emoji} **[{finding.priority.value.upper()}]** {finding.description}")
            if finding.tool_name:
                lines.append(f"   - Herramienta: `{finding.tool_name}`")
            lines.append(f"   - Contexto: {finding.context}")
            if finding.evidence:
                lines.append(f"   - Evidencia: {finding.evidence[0][:100]}...")
            lines.append("")

        lines.extend([
            "### Propuestas",
            "",
        ])

        for i, (proposal, pid) in enumerate(zip(analysis.proposals, proposal_ids)):
            emoji = priority_emoji.get(proposal.get("priority", "medium"), "⚪")
            lines.extend([
                f"**#{i+1}** {emoji} (`{proposal.get('priority', 'medium').upper()}`)",
                "",
                f"- **Título**: {proposal.get('title', '')}",
                f"- **Descripción**: {proposal.get('description', '')}",
                f"- **Cambio**: `{proposal.get('change_location', '')}`",
                f"- **Contenido**: ```{proposal.get('change_content', '')}```",
                f"- **Razón**: {proposal.get('reason', '')}",
                "",
                f"[id: {pid}] [apply] [dismiss] [edit]",
                "",
                "---",
                "",
            ])

        filename.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote proposals to %s", filename)

    def format_pending_proposals(self, proposals: list[DreamProposal]) -> str:
        """Format pending proposals for display."""
        if not proposals:
            return "No hay propuestas pendientes de sueños. Ejecuta `/dreams` cuando tengas nuevas interacciones."

        priority_emoji = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
        }

        lines = ["## 💭 Propuestas de Sueños", ""]

        current_date = None
        for p in proposals:
            if p.date != current_date:
                current_date = p.date
                lines.append(f"### 📅 {p.date}")
                lines.append("")

            emoji = priority_emoji.get(p.priority.value, "⚪")
            lines.extend([
                f"**#{p.proposal_index}** {emoji} `{p.priority.value.upper()}`",
                f"**{p.title}**",
                "",
                f"{p.description}",
                "",
                f"`{p.change_location}`",
                f"```",
                f"{p.change_content}",
                f"```",
                "",
                f"_Razón: {p.reason}_",
                "",
                f"[id: {p.id}] /dreams apply {p.id} | /dreams dismiss {p.id} | /dreams edit {p.id}",
                "",
                "---",
                "",
            ])

        return "\n".join(lines)

    def get_proposal_markdown(self, proposal: DreamProposal) -> str:
        """Get markdown for a single proposal."""
        priority_emoji = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
        }
        emoji = priority_emoji.get(proposal.priority.value, "⚪")

        return f"""## 💭 Propuesta #{proposal.proposal_index} - {proposal.date}

{emoji} **{proposal.priority.value.upper()}**

### {proposal.title}

{proposal.description}

**Ubicación del cambio:** `{proposal.change_location}`

```
{proposal.change_content}
```

**Razón:** {proposal.reason}

---
[apply] /dreams apply {proposal.id}
[dismiss] /dreams dismiss {proposal.id}
[edit] /dreams edit {proposal.id}
"""