"""Tests for eyetor.chat.session._TOOL_INTENT_RE.

Regression coverage for an announce-without-call miss: the old pattern only
matched an enumerated verb list ("voy a ejecutar/llamar/usar/..."), so
"Voy a buscar información..." slipped through, the nudge never fired, and the
model ended its turn with an unfulfilled promise. The Spanish patterns are now
generic ("voy a <any verb>"); _is_asking_user remains the guard that keeps
legitimate requests for user input from being nudged.
"""

from __future__ import annotations

from eyetor.chat.session import _TOOL_INTENT_RE, _is_asking_user


def _announces(text: str) -> bool:
    return bool(_TOOL_INTENT_RE.search(text))


# ── Announcements: any first-person action verb must match ────────────


class TestSpanishAnnouncements:
    def test_telegram_incident(self):
        """Exact reproduction of the shopping-list incident."""
        text = (
            "Voy a buscar información sobre dónde puedes comprar "
            '"cafe en grano" para ver las mejores opciones.'
        )
        assert _announces(text) is True

    def test_voy_a_any_verb(self):
        assert _announces("Voy a comprobar la lista de la compra.") is True

    def test_vamos_a(self):
        assert _announces("Vamos a revisar el estado del servicio.") is True

    def test_procedo_a(self):
        assert _announces("Procedo a revisar el log del sistema.") is True

    def test_ahora_mismo_lo(self):
        assert _announces("Ahora mismo lo miro.") is True

    def test_ahora_lo(self):
        assert _announces("Ahora lo compruebo y te digo.") is True

    def test_dejame(self):
        assert _announces("Déjame buscar eso un momento.") is True

    def test_dejame_unaccented(self):
        assert _announces("dejame mirar la agenda") is True

    def test_legacy_future_tense(self):
        assert _announces("Ejecutaré el comando en cuanto pueda.") is True


class TestEnglishAnnouncements:
    def test_let_me(self):
        assert _announces("Let me try fetching that page again.") is True

    def test_ill_call(self):
        assert _announces("I'll call the search tool now.") is True


# ── Non-announcements: plain answers must NOT match ───────────────────


class TestPlainAnswers:
    def test_final_answer(self):
        assert _announces("Tienes un producto en la lista: café en grano.") is False

    def test_past_tense_report(self):
        assert _announces("He añadido el producto a tu lista.") is False

    def test_english_plain(self):
        assert _announces("Your list has one item: coffee beans.") is False


# ── Guard interaction: asking the user suppresses the nudge ───────────


class TestAskingUserGuard:
    def test_announcement_with_question_is_asking(self):
        text = "Voy a buscar tiendas. ¿En qué ciudad vives?"
        assert _announces(text) is True
        assert _is_asking_user(text) is True

    def test_announcement_without_question_is_not_asking(self):
        text = "Voy a buscar las mejores opciones de compra."
        assert _announces(text) is True
        assert _is_asking_user(text) is False
