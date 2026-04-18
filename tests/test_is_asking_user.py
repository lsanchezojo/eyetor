"""Tests for eyetor.chat.session._is_asking_user.

Regression coverage for a false-positive in the announce-without-call nudge
gate: a '?' inside a URL query string (e.g. `?item=10523`) or inside a code
span was being treated as a question to the user, which suppressed the nudge
and let the model end its turn with an unfulfilled announcement.
"""

from __future__ import annotations

from eyetor.chat.session import _is_asking_user


# ── Real questions: must still be classified as questions ─────────────


class TestRealQuestions:
    def test_trailing_question_mark(self):
        assert _is_asking_user("¿prefieres la opción A o la B?") is True

    def test_english_question(self):
        assert _is_asking_user("Which file should I edit?") is True

    def test_marker_indicame(self):
        assert _is_asking_user("Indícame la ruta del archivo.") is True

    def test_marker_please_provide(self):
        assert _is_asking_user("Please provide the API key.") is True


# ── Regressions: '?' inside URL or code must NOT count as a question ──


class TestUrlAndCodeRegressions:
    def test_query_string_alone(self):
        text = "Voy a intentar acceder a http://example.com/path?foo=bar ahora."
        assert _is_asking_user(text) is False

    def test_url_in_backticks_with_announcement(self):
        """Exact reproduction of the Telegram incident."""
        text = (
            "Voy a intentar una última vez acceder a esa URL específica "
            "(`https://masqueoca.com/tienda/producto.asp?item=10523`) "
            "pidiéndole al browser que me entregue más texto. "
            "Procedo con la revisión de la página de producto."
        )
        assert _is_asking_user(text) is False

    def test_fenced_code_with_question_mark(self):
        text = (
            "Aquí tienes el ejemplo:\n"
            "```python\n"
            "url = 'https://api.example.com/v1/items?limit=10'\n"
            "```\n"
            "Procedo con la consulta."
        )
        assert _is_asking_user(text) is False

    def test_inline_code_with_question_mark(self):
        text = "El patrón `foo?bar` es literal. Voy a ejecutarlo."
        assert _is_asking_user(text) is False


# ── Mixed cases: real question wins even when a URL also contains '?' ─


class TestMixed:
    def test_url_plus_real_question(self):
        text = (
            "He consultado http://example.com/path?foo=bar. "
            "¿Quieres que continúe o prefieres otra fuente?"
        )
        assert _is_asking_user(text) is True

    def test_empty_string(self):
        assert _is_asking_user("") is False

    def test_none_like_no_marks(self):
        assert _is_asking_user("Respuesta final sin preguntas.") is False
