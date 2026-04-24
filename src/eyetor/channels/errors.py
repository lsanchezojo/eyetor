"""User-facing error formatting for chat channels.

Channels (Telegram, CLI, ...) call ``format_user_error(exc)`` instead of
piping raw stack-trace fragments to end users. Mappings are intentionally
short and actionable in Spanish, since this project ships with Spanish UX.
"""

from __future__ import annotations

import httpx

from eyetor.providers.base import ContextOverflowError


def format_user_error(exc: BaseException) -> str:
    """Map a known exception to a friendly message.

    Falls back to a generic, non-technical line for unknown errors so the
    user never sees a stack-trace fragment.
    """
    if isinstance(exc, ContextOverflowError):
        return (
            "La conversación se ha vuelto demasiado larga para el modelo. "
            "Usa /reset para empezar limpio o reformula la pregunta de forma más concisa."
        )
    if isinstance(exc, RuntimeError) and "All providers" in str(exc):
        return (
            "Todos los modelos están fallando ahora mismo. "
            "Reintenta en unos minutos o usa /model para cambiar de proveedor."
        )
    if isinstance(exc, httpx.TimeoutException):
        return (
            "El modelo está tardando demasiado en responder. "
            "Reintenta o usa /model para cambiar a un proveedor más rápido."
        )
    if isinstance(exc, httpx.ConnectError):
        return (
            "No se puede conectar con el proveedor. "
            "Comprueba la red o usa /model para probar otro."
        )
    return f"Ha ocurrido un error técnico ({type(exc).__name__}). Si persiste, revisa los logs."
