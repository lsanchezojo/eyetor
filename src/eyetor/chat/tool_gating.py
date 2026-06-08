"""Keyword gating that decides which conditional tool groups a turn needs.

Token-heavy, rarely-used tool schemas (scheduler, image, kb, install) are
tagged with a ``group`` on their ``ToolDefinition`` and only sent to the
model on turns whose text triggers them. Always-on tools (skills, memory,
delegate) carry ``group=None`` and are never gated — delegate is kept always
on because its system-prompt instructions are unconditional.

Matching is accent- and case-insensitive and involves no LLM call, so it
adds no latency. The triggers are deliberately generous: over-including a
group only costs a few prompt tokens, while under-including it would make a
capability silently unavailable. Follow-up turns ("cancélalo", "sí, hazlo")
are handled by the session's sticky/confirmation logic, not here.
"""

from __future__ import annotations

import re
import unicodedata

# Trigger patterns are matched against accent-stripped, lowercased text.
_GROUP_PATTERNS: dict[str, re.Pattern[str]] = {
    "scheduler": re.compile(
        r"\b("
        r"record(ar|atorio|aro)|recuerda|recuerdame|"
        r"program(a|ar|ame|acion)|agend(a|ar|ame)|"
        r"tarea|alarma|cron|avisame|aviso|notifica(me|r)?|"
        r"todos\s+los|cada\s+\d+\s*(m|h|d|min|hora|dia)|"
        r"cada\s+(dia|semana|mes|hora|lunes|martes|miercoles|jueves|viernes|sabado|domingo|\d)|"
        r"every\s+\d|schedule|reminder|"
        r"manana|pasado\s+manana|"
        r"(lunes|martes|miercoles|jueves|viernes|sabado|domingo)|"
        r"a\s+las\s+\d"
        r")\b"
    ),
    "image": re.compile(
        r"\b("
        r"(gener|cre|dibuj|haz|hazme|pint|render|disen)\w*\s+(una?\s+)?"
        r"(imagen|imagenes|foto|dibujo|ilustracion|logo)|"
        r"imagina(te)?\s+(una?|el|la)\s|"
        r"draw|render|picture|image\s+of"
        r")\b"
    ),
    "install": re.compile(
        r"\b("
        r"instal(a|ar|ame|acion)|install|"
        r"paquete|package|"
        r"command\s+not\s+found|no\s+esta\s+instalad|"
        r"falta\s+el\s+(comando|programa|binario|paquete)"
        r")\b"
    ),
    "kb": re.compile(
        r"\b("
        r"busca\s+en\s+(los?\s+|la\s+)?(docs|documentos|documentacion)|"
        r"segun\s+(el|la|los|las)\s+(documento|documentacion|manual|guia|nota)|"
        r"base\s+de\s+conocimiento|knowledge\s*base|kb_|"
        r"que\s+dice\s+(el|la)\s+(documento|manual|guia|pdf|nota)|"
        r"en\s+la\s+documentacion"
        r")\b"
    ),
}

# All gating groups known to the system. Used to validate tool tags and as
# the universe when gating is disabled.
KNOWN_GROUPS: frozenset[str] = frozenset(_GROUP_PATTERNS)


def _normalize(text: str) -> str:
    """Lowercase and strip combining accents so 'recuérdame' == 'recuerdame'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def select_groups(text: str) -> set[str]:
    """Return the conditional tool groups whose triggers appear in ``text``."""
    norm = _normalize(text or "")
    return {group for group, pat in _GROUP_PATTERNS.items() if pat.search(norm)}
