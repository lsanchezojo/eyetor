"""Tests for eyetor.chat.session._is_scaffolding_leak.

Safety net for the "Boteando" group incident: when the announce-without-call
nudge fails to make the model emit a tool_call, a small model sometimes replies
*to the nudge* — apologizing and narrating its own tool/plan process instead of
answering. That meta-commentary must never be sent as the final answer. The
detector requires BOTH an apology/error admission AND an internal-process
reference so a normal answer that merely mentions a tool does not trip it.
"""

from __future__ import annotations

from eyetor.chat.session import _is_scaffolding_leak


# ── Must be detected as a leak ────────────────────────────────────────


class TestLeaks:
    def test_boteando_incident_verbatim(self):
        """Exact reproduction of the group incident that motivated the guard."""
        text = (
            "Lo siento mucho, tienes razón. He cometido un error en la ejecución "
            "del plan y en la comunicación de las acciones. Dado que ya he "
            "realizado las búsquedas necesarias y he proporcionado la respuesta "
            "detallada sobre la marca BSEED, no necesito realizar más llamadas a "
            "herramientas en este momento. Si necesitas que profundice en algún "
            "punto específico del análisis o que realice alguna otra consulta, "
            "dímelo y lo haré de inmediato."
        )
        assert _is_scaffolding_leak(text) is True

    def test_apology_plus_toolcall_term(self):
        text = "Mis disculpas, no emití la tool_call correctamente."
        assert _is_scaffolding_leak(text) is True

    def test_english_apology_plus_process(self):
        text = "I'm sorry, you're right — I won't make more tool_calls now."
        assert _is_scaffolding_leak(text) is True


# ── Must NOT be detected (legitimate answers) ─────────────────────────


class TestNonLeaks:
    def test_legit_tool_mention(self):
        text = (
            "Usé la búsqueda web y encontré que BSEED es una marca de domótica; "
            "sus enchufes inteligentes son compatibles con Tuya."
        )
        assert _is_scaffolding_leak(text) is False

    def test_question_to_user(self):
        text = "¿Quieres que busque los precios en Amazon o en AliExpress?"
        assert _is_scaffolding_leak(text) is False

    def test_apology_without_process(self):
        text = "Lo siento, no tengo esa información ahora mismo."
        assert _is_scaffolding_leak(text) is False

    def test_process_without_apology(self):
        text = "He ejecutado la búsqueda y ya no necesito más herramientas."
        assert _is_scaffolding_leak(text) is False

    def test_empty_string(self):
        assert _is_scaffolding_leak("") is False
