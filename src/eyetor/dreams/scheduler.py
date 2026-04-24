"""Dreams scheduler — integrates with SchedulerChannel for periodic dream analysis."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from apscheduler.triggers.cron import CronTrigger

from eyetor.config import VectorConfig
from eyetor.dreams.analyzer import DreamsAnalyzer
from eyetor.dreams.config import DreamConfig
from eyetor.dreams.proposer import ProposalGenerator
from eyetor.dreams.store import DreamsStore, ProposalStatus
from eyetor.scheduler.channel import SchedulerChannel

logger = logging.getLogger(__name__)


class DreamsScheduler:
    """Manages scheduled dream analysis runs."""

    _MAX_CONSECUTIVE_FAILURES = 3

    def __init__(
        self,
        config: VectorConfig,
        scheduler: SchedulerChannel,
    ) -> None:
        self._config = config
        self._scheduler = scheduler
        self._dreams_config = config.dreams
        self._store: DreamsStore | None = None
        self._consecutive_failures = 0

    def _get_store(self) -> DreamsStore:
        """Lazy initialization of store."""
        if self._store is None:
            db_path = Path("~/.eyetor/dreams.db").expanduser()
            self._store = DreamsStore(db_path)
        return self._store

    async def run_dream_analysis(self) -> str:
        """Run dream analysis and return status message.

        Wraps the full pipeline (config access + analyzer + generator) in a
        single guard so that failures during initialization (e.g. stale
        in-memory pydantic models after a hot-edit) don't escape to APScheduler
        and pollute the log every night. After ``_MAX_CONSECUTIVE_FAILURES``
        runs, the job is paused so it stops trying until the operator restarts
        the service or resumes the job manually.
        """
        logger.info("Starting dream analysis cycle")
        try:
            message = await self._run_dream_analysis_inner()
        except Exception as exc:
            self._consecutive_failures += 1
            logger.exception(
                "Dream analysis failed (consecutive failures: %d/%d)",
                self._consecutive_failures,
                self._MAX_CONSECUTIVE_FAILURES,
            )
            if self._consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
                self._pause_job_after_repeated_failures()
            return f"Error en análisis de sueños: {type(exc).__name__}: {exc}"
        else:
            self._consecutive_failures = 0
            return message

    async def _run_dream_analysis_inner(self) -> str:
        if self._dreams_config is None:
            return "Sistema de sueños deshabilitado."

        store = self._get_store()
        config = self._dreams_config

        sessions_dir = Path("~/.eyetor/sessions").expanduser()
        tracking_db = Path(config.tracking.db_path.replace("~", str(Path.home())))
        memory_db = Path(config.memory_db_path.replace("~", str(Path.home())))

        analyzer = DreamsAnalyzer(
            store=store,
            sessions_dir=sessions_dir,
            tracking_db=tracking_db,
            memory_db=memory_db,
            config=config,
        )

        generator = ProposalGenerator(
            store=store,
            output_dir=Path(config.output_dir).expanduser(),
        )

        analysis = await analyzer.run_analysis()
        if analysis.findings:
            proposal_ids = generator.generate_and_save(analysis)
            return f"Análisis de sueños completado. {len(proposal_ids)} propuesta(s) generada(s)."
        return "Análisis de sueños completado. No se encontraron hallazgos significativos."

    def _pause_job_after_repeated_failures(self) -> None:
        """Pause the dream_analysis APScheduler job until manually resumed."""
        aps = getattr(self._scheduler, "_scheduler", None)
        if aps is None:
            return
        try:
            aps.pause_job("dream_analysis")
            logger.error(
                "Dream analysis paused after %d consecutive failures. "
                "Reinicia el servicio o reanuda el job manualmente para reactivarlo.",
                self._consecutive_failures,
            )
        except Exception as exc:
            logger.warning("Could not pause dream_analysis job: %s", exc)

    def schedule_dreams(self) -> None:
        """Schedule the dream analysis job as a direct APScheduler callable."""
        if not self._dreams_config.enabled:
            logger.info("Dreams system disabled")
            return

        schedule = self._dreams_config.schedule

        try:
            trigger = CronTrigger.from_crontab(schedule)
        except Exception as e:
            logger.warning("Invalid cron schedule %s: %s", schedule, e)
            return

        try:
            self._scheduler.add_callable_job(
                job_id="dream_analysis",
                func=self.run_dream_analysis,
                trigger=trigger,
                name="Sueños - Análisis Nocturno",
            )
            logger.info("Dream analysis scheduled: %s", schedule)
        except Exception as e:
            logger.warning("Failed to schedule dreams: %s", e)

    async def handle_apply(self, proposal_id: int) -> str:
        """Handle apply proposal command."""
        store = self._get_store()
        store.update_proposal_status(proposal_id, ProposalStatus.APPLIED)
        return f"Propuesta #{proposal_id} marcada como aplicada."

    async def handle_dismiss(self, proposal_id: int) -> str:
        """Handle dismiss proposal command."""
        store = self._get_store()
        store.update_proposal_status(proposal_id, ProposalStatus.DISMISSED)
        return f"Propuesta #{proposal_id} descartada."

    async def handle_list(self) -> str:
        """Handle list proposals command."""
        store = self._get_store()
        generator = ProposalGenerator(store, Path(self._dreams_config.output_dir).expanduser())

        proposals = store.get_pending_proposals()
        return generator.format_pending_proposals(proposals)

    async def handle_run(self) -> str:
        """Handle manual run command."""
        return await self.run_dream_analysis()