from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from eyetor.scheduler.channel import SchedulerChannel, _parse_trigger
from eyetor.scheduler.store import ScheduledTask, SchedulerStore

TZ = "Europe/Madrid"


@pytest.mark.parametrize(
    "expr,delta",
    [
        ("in 30 seconds", timedelta(seconds=30)),
        ("in 10m", timedelta(minutes=10)),
        ("en 2 horas", timedelta(hours=2)),
        ("dentro de 5 minutos", timedelta(minutes=5)),
    ],
)
def test_relative_offset_is_one_shot(expr, delta):
    from apscheduler.triggers.date import DateTrigger

    trigger = _parse_trigger(expr, TZ)
    assert isinstance(trigger, DateTrigger)
    expected = datetime.now(ZoneInfo(TZ)) + delta
    assert abs((trigger.run_date - expected).total_seconds()) < 5


def _task(schedule: str) -> ScheduledTask:
    return ScheduledTask(name="t", prompt="p", schedule=schedule, session_id="s", notify="none")


def test_add_task_rejects_unparseable_schedule_without_persisting(tmp_path):
    store = SchedulerStore(str(tmp_path / "s.db"))
    channel = SchedulerChannel(store=store, session_manager=None, bot_token=None)
    with pytest.raises(ValueError):
        channel.add_task(_task("whenever you like"))
    assert store.list_all() == []


def test_add_task_persists_relative_one_shot_as_absolute(tmp_path):
    store = SchedulerStore(str(tmp_path / "s.db"))
    channel = SchedulerChannel(store=store, session_manager=None, bot_token=None)
    channel.add_task(_task("in 30 seconds"))
    (saved,) = store.list_all()
    assert saved.schedule[:4].isdigit()  # 'YYYY-MM-DD HH:MM:SS'
    assert _parse_trigger(saved.schedule, TZ).run_date > datetime.now(ZoneInfo(TZ))


def test_scheduled_prompt_marks_it_as_firing_task():
    from eyetor.scheduler.channel import _scheduled_prompt

    text = _scheduled_prompt(_task("in 1 minute"))
    assert "Tarea programada «t»" in text
    assert text.endswith("Contenido de la tarea: p")


def test_one_shot_removed_after_firing_recurring_kept(tmp_path):
    import asyncio

    class _Session:
        def reset(self):
            pass

        async def send_sync(self, prompt):
            return "ok"

    class _Mgr:
        def get_or_create(self, session_id):
            return _Session()

    store = SchedulerStore(str(tmp_path / "s.db"))
    channel = SchedulerChannel(store=store, session_manager=_Mgr(), bot_token=None)
    one = channel.add_task(_task("in 1 minute"))
    rec = channel.add_task(_task("every 30m"))
    asyncio.run(channel._run_task(one.id))
    asyncio.run(channel._run_task(rec.id))
    assert [t.id for t in store.list_all()] == [rec.id]
