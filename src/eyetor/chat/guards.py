"""Reply heuristics for ChatSession.send(): what the model's text implies.

Detects announced-but-not-emitted tool calls, questions to the user,
scaffolding leaks, user confirmations, and claims ("he encendido…",
"te lo recordaré") with no tool run to back them; plus the nudge texts
send() injects to recover from each case.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from eyetor.models.messages import Message
from eyetor.models.tools import ToolDefinition


# Phrases a model may emit when it *announces* a tool call but forgets to
# emit the structured tool_call in the same turn. Used to decide whether
# to nudge it once before accepting the text as the final answer.
#
# The Spanish patterns are deliberately generic ("voy a <any verb>") rather
# than a verb list: enumerations stay incomplete ("voy a buscar" once slipped
# through) and a false positive costs at most one extra LLM call per turn —
# the nudge only fires when tools are available, nothing has executed yet,
# and the message is not a question to the user (_is_asking_user).
_TOOL_INTENT_RE = re.compile(
    r"((voy|vamos) a \w+"
    r"|procedo a \w+"
    r"|paso a \w+"
    r"|d[ée]jame \w+"
    r"|ahora (mismo )?lo \w+"
    r"|(ejecutar|llamar|usar|lanzar|invocar)(é| ahora| la herramienta)"
    r"|intentar(é| de nuevo| otra vez| nuevamente)"
    r"|reintent\w+"
    r"|probar(é)? (de nuevo|otra vez)"
    r"|let me (try|call|retry|invoke|use|run|execute)"
    r"|i'?ll (try|call|retry|invoke|use|run|execute)"
    r"|i will (try|retry|call|invoke|run|execute)"
    r"|retrying|trying again"
    r"|now (i'?ll|let me|i will) (call|use|run|execute|invoke))",
    re.IGNORECASE,
)


# First-person past-tense claims of a smart-home action ("he apagado la luz").
# A small model sometimes reads the state with GetLiveContext and then *says*
# it acted without emitting the action call (incident 2026-10-30: "He apagado
# la luz del salón" with no HassTurnOff). Only "he/hemos + participle" forms:
# "ya está apagada" is a legitimate answer when nothing needed doing.
_HOME_ACTION_CLAIM_RE = re.compile(
    r"\b(he|hemos)\s+(\w+\s+)?"
    r"(apagad|encendid|prendid|activad|desactivad|abiert|cerrad|subid|bajad|"
    r"puest|ajustad|cambiad|pausad|reanudad|silenciad|iniciad|lanzad|"
    r"enviad|dispensad|regulad|bloquead|desbloquead)[oa]s?\b",
    re.IGNORECASE,
)

# Read-only tools in the "home" group; anything else in that group acts on the house.
_HOME_READ_ONLY_SUFFIXES = ("GetLiveContext", "GetDateTime")


def _is_home_action_tool(tool: ToolDefinition | None) -> bool:
    return (
        tool is not None
        and tool.group == "home"
        and not tool.name.endswith(_HOME_READ_ONLY_SUFFIXES)
    )


# "Lo he guardado", "¡Apuntado!", "lo recordaré", "tomo nota"… (not "no he guardado").
_MEMORY_CLAIM_RE = re.compile(
    r"(?<!no )\b(he|hemos)\s+(\w+\s+)?"
    r"(guardad|apuntad|anotad|memorizad|registrad|almacenad)[oa]s?\b"
    r"|^\W*(guardad|apuntad|anotad|memorizad)[oa]\b"
    r"|(?<!te )\b(lo|la|los|las)\s+recordaré\b"
    r"|\btomo\s+nota\b|\bnota\s+tomada\b|\bme\s+lo\s+apunto\b"
    r"|\blo\s+tendré\s+en\s+cuenta\b"
    r"|\bqueda(n)?\s+(guardad|apuntad|anotad|registrad)[oa]s?\b",
    re.IGNORECASE | re.MULTILINE,
)

# "Te avisaré", "te lo recordaré", "he programado un recordatorio", "alarma puesta"…
_REMINDER_CLAIM_RE = re.compile(
    r"\bte\s+(lo\s+|la\s+)?(avisaré|recordaré)\b"
    r"|(?<!no )\b(he|hemos)\s+(\w+\s+)?(programad|cread|configurad|puest|añadid)[oa]s?"
    r"\s+(\w+\s+)?(recordatorio|aviso|alarma|alerta)s?\b"
    r"|\b(recordatorio|aviso|alarma|alerta)\s+(creado|programado|configurado|puesto|listo)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _ClaimGuard:
    """An action the model may claim in text without having called its tool.

    The guard applies on turns where an ``offered`` tool was sent; a successful
    call of a ``done_by`` tool this turn backs the claim."""

    name: str
    claim_re: re.Pattern[str]
    offered: Callable[[ToolDefinition], bool]
    done_by: Callable[[ToolDefinition | None], bool]
    nudge: str
    warning: str


def _tool_failed(result: str) -> bool:
    head = result.lstrip()[:80]
    return head.startswith(("Error", '{"error"')) or '"ok": false' in head


# Markers that indicate the model is requesting information from the user
# rather than forgetting to emit a tool_call. A message with any of these
# should NOT trigger the announce-without-call nudge: the model is legitimately
# blocked on missing parameters and correctly chose not to emit a call.
_ASK_USER_MARKERS = (
    "indícame",
    "indicame",
    "dime la",
    "dime el",
    "dime dónde",
    "dime donde",
    "proporcióname",
    "proporcioname",
    "necesito que me",
    "por favor indica",
    "por favor dime",
    "por favor proporciona",
    "please provide",
    "please tell me",
    "could you tell",
    "could you provide",
    "let me know",
)


# Patterns stripped before scanning for question marks in `_is_asking_user`.
# Without these, a '?' inside a URL query string (e.g. `?item=10523`) or a
# code example makes the heuristic misclassify an announcement as a question
# and suppresses the announce-without-call nudge.
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def _is_asking_user(text: str) -> bool:
    """True if the model's message is a request for user input.

    Fenced/inline code and URLs are stripped before scanning for '?'/'¿' so
    a query string like ``?item=10523`` or a code example doesn't count as
    a question to the user. The ``_ASK_USER_MARKERS`` check still runs on
    the original text because those are prose phrases.
    """
    if not text:
        return False
    cleaned = _FENCED_CODE_RE.sub("", text)
    cleaned = _INLINE_CODE_RE.sub("", cleaned)
    cleaned = _URL_RE.sub("", cleaned)
    if "?" in cleaned or "¿" in cleaned:
        return True
    lowered = text.lower()
    return any(m in lowered for m in _ASK_USER_MARKERS)


# When the announce-without-call nudge does NOT get the model to emit a
# tool_call, a small model sometimes replies *to the nudge* as if it were user
# criticism — apologizing and narrating its own tool/plan process instead of
# answering. That meta-commentary then leaks to the user as the final answer
# (see the "Boteando" group incident: "Lo siento, tienes razón... no necesito
# realizar más llamadas a herramientas"). We detect it by requiring BOTH an
# apology/error admission AND a reference to the model's internal process, so a
# legitimate answer that merely mentions a tool ("usé la búsqueda y encontré…")
# does not trip the guard.
_SCAFFOLD_APOLOGY_RE = re.compile(
    r"(lo siento"
    r"|mis disculpas"
    r"|disculpa(s|me)?\b"
    r"|perd[óo]n"
    r"|tienes raz[óo]n"
    r"|he cometido (un|el) error"
    r"|me he equivocado"
    r"|i (apologize|'?m sorry)"
    r"|you'?re right"
    r"|my mistake)",
    re.IGNORECASE,
)
_SCAFFOLD_PROCESS_RE = re.compile(
    r"(tool[_ ]?calls?"
    r"|llamadas? a (la |las )?herramientas?"
    r"|llamar a (la |las )?herramientas?"
    r"|ejecuci[óo]n del plan"
    r"|comunicaci[óo]n de (las|mis) acciones"
    r"|no necesito (realizar|hacer|llamar a|usar)\w* (m[áa]s )?(llamadas|herramientas)"
    r"|no (necesito|hace falta) (realizar|hacer|usar)\w* m[áa]s (llamadas|herramientas)"
    r"|en la ejecuci[óo]n"
    r"|mi proceso interno)",
    re.IGNORECASE,
)


def _is_scaffolding_leak(text: str) -> bool:
    """True if ``text`` is the model talking *about its own tool/plan process*
    (with an apology) instead of answering the user.

    Requires co-occurrence of an apology/error admission and an internal-process
    reference to keep false positives bounded: a normal answer that mentions a
    tool in passing won't match, but a meta-reply to the announce-without-call
    nudge ("lo siento, tienes razón... no necesito más llamadas a herramientas")
    will. Used as a safety net so such replies are never sent as the final
    answer. See ``_TOOL_INTENT_RE`` for the announce-without-call nudge itself.
    """
    if not text:
        return False
    return bool(_SCAFFOLD_APOLOGY_RE.search(text)) and bool(
        _SCAFFOLD_PROCESS_RE.search(text)
    )


_CONFIRMATION_RE = re.compile(
    r"^(s[ií]|ok|dale|hazlo|adelante|confirmo|confirma|vale|"
    r"venga|va|claro|por supuesto|yes|yeah|yep|sure|go ahead|do it"
    r")[\s.!,]*$",
    re.IGNORECASE,
)


def _is_user_confirmation(text: str) -> bool:
    """True if ``text`` is a short affirmative confirmation."""
    return bool(_CONFIRMATION_RE.match(text.strip()))


_ACTION_PROPOSAL_RE = re.compile(
    r"\b("
    r"comando|command|herramienta|tool|ejecut\w*|lanz\w*|corr\w*|"
    r"run|execute|launch|dispatch|cerr\w*|abr\w*|borr\w*|elimin\w*|"
    r"instal\w*|reinici\w*|reload"
    r")\b",
    re.IGNORECASE,
)


def _last_assistant_proposed_action(messages: list[Message]) -> bool:
    """True if the last assistant turn asked to confirm an actionable step."""
    for msg in reversed(messages):
        if msg.role == "assistant":
            content = msg.content or ""
            return (
                bool(content.strip())
                and _is_asking_user(content)
                and bool(_ACTION_PROPOSAL_RE.search(content))
            )
        if msg.role == "user":
            return False
    return False


# Instructions injected into the turn by send()'s recovery paths. They are
# addressed to the model, never shown to the user.
_NUDGE_CONFIRMED_ACTION = (
    "El usuario acaba de confirmar la acción que propusiste. "
    "DEBES ejecutarla AHORA emitiendo la tool_call estructurada "
    "correspondiente en esta misma respuesta. No respondas solo "
    "con texto: si hay una herramienta adecuada, llama a la "
    "herramienta con los parámetros correctos."
)
_NUDGE_INVALID_TOOL_ARGS = (
    "La tool_call anterior tenía arguments que no eran JSON "
    "válido y se ha descartado. Vuelve a intentarlo ahora "
    "emitiendo una tool_call estructurada válida. No construyas "
    "JSON dentro de strings largos si existe una herramienta con "
    "parámetros estructurados."
)
_NUDGE_EMPTY_REPLY = (
    "Recordatorio del sistema: aún no hay respuesta visible ni tool_call "
    "estructurada. Continúa ahora: si necesitas actuar en el ordenador, "
    "emite la tool_call correcta; si no necesitas herramientas, responde "
    "directamente a la consulta del usuario en español, sin hablar de tu "
    "proceso interno y sin disculparte. No dejes la respuesta vacía."
)
_NUDGE_HOME_CLAIM = (
    "Recordatorio del sistema: has dicho que hiciste una acción en la casa, "
    "pero en este turno no se ha ejecutado ninguna herramienta de acción, así "
    "que no ha cambiado nada. Emite ahora la tool_call estructurada (p. ej. "
    "intent__HassTurnOff / intent__HassTurnOn con el nombre exacto de un "
    "dispositivo). Solo si en realidad no hace falta ninguna acción, responde "
    "sin afirmar que hayas hecho nada."
)
_HOME_CLAIM_WARNING = (
    "\n\n⚠️ Ojo: no se ha llegado a ejecutar ninguna acción en "
    "Home Assistant, así que puede que no haya cambiado nada."
)

# Checked in order; the first claim that matches wins. Reminders go before
# memory so "te lo recordaré" counts as a reminder.
_CLAIM_GUARDS: tuple[_ClaimGuard, ...] = (
    _ClaimGuard(
        name="home",
        claim_re=_HOME_ACTION_CLAIM_RE,
        offered=_is_home_action_tool,
        done_by=_is_home_action_tool,
        nudge=_NUDGE_HOME_CLAIM,
        warning=_HOME_CLAIM_WARNING,
    ),
    _ClaimGuard(
        name="reminder",
        claim_re=_REMINDER_CLAIM_RE,
        offered=lambda t: t.name == "schedule_task",
        done_by=lambda t: t is not None and t.name == "schedule_task",
        nudge=(
            "Recordatorio del sistema: has dicho que programaste un recordatorio, "
            "pero en este turno no se ha ejecutado schedule_task, así que no existe "
            "ninguno. Emite ahora la tool_call estructurada de schedule_task. Solo si "
            "no hace falta ningún recordatorio, responde sin afirmar que lo hayas "
            "programado."
        ),
        warning=(
            "\n\n⚠️ Ojo: no se ha llegado a programar ningún recordatorio, "
            "así que no te avisaré."
        ),
    ),
    _ClaimGuard(
        name="memory",
        claim_re=_MEMORY_CLAIM_RE,
        offered=lambda t: t.name == "remember",
        # Lists, notes and files are stored by other tools: any tool run backs it.
        done_by=lambda t: t is not None,
        nudge=(
            "Recordatorio del sistema: has dicho que guardaste o apuntaste algo, "
            "pero en este turno no se ha ejecutado ninguna herramienta, así que no "
            "se ha guardado nada. Emite ahora la tool_call estructurada (remember "
            "para datos del usuario, o la herramienta que toque para listas, notas "
            "o ficheros). Solo si no hay nada que guardar, responde sin afirmar que "
            "hayas guardado nada."
        ),
        warning="\n\n⚠️ Ojo: no se ha llegado a guardar nada, así que puede que no lo recuerde.",
    ),
)


def _unbacked_claim(
    content: str, tool_defs: list[ToolDefinition] | None, backed: set[str]
) -> _ClaimGuard | None:
    """The first guard whose action is claimed in ``content`` but never ran."""
    if not content or not tool_defs:
        return None
    for guard in _CLAIM_GUARDS:
        if (
            guard.name not in backed
            and any(guard.offered(t) for t in tool_defs)
            and guard.claim_re.search(content)
        ):
            return guard
    return None
_NUDGE_ANNOUNCED_CALL = (
    "Recordatorio del sistema: para usar una herramienta debes emitir la "
    "tool_call estructurada; describirla en texto no la ejecuta. Si necesitas "
    "una herramienta, emítela ahora con los parámetros correctos. Si no, "
    "responde directamente a la consulta del usuario en español, sin hablar "
    "de herramientas ni de tu proceso interno y sin disculparte."
)
_FORCE_SYNTHESIS = (
    "Sintetiza ahora una respuesta clara y breve al usuario "
    "usando ÚNICAMENTE la información obtenida en las herramientas "
    "anteriores. No llames más herramientas. Responde en lenguaje "
    "natural, en español."
)
_FORCE_CLEAN_ANSWER = (
    "Responde directamente a la consulta original del usuario "
    "en español. No menciones herramientas, llamadas, tool_calls, "
    "planes ni tu proceso interno; no te disculpes. Si ya tienes la "
    "información, da la respuesta; si te falta un dato, pídelo de "
    "forma concreta."
)
_FORCE_BREAK_LOOP = (
    "IMPORTANTE: estás repitiendo las mismas llamadas a herramientas en bucle. "
    "Da ahora tu respuesta final al usuario, en español, con la información que "
    "ya tienes. NO llames a más herramientas."
)
