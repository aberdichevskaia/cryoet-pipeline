"""scripts/litellm_pacer.py: no 60 s window holds more requests than the limit."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "litellm_pacer.py"


def load_pacer():
    spec = importlib.util.spec_from_file_location("litellm_pacer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["litellm_pacer"] = module
    spec.loader.exec_module(module)
    return module


pacer = load_pacer()


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_free_places_go_at_once():
    clock = Clock()
    window = pacer.Window(3, clock)
    for _ in range(3):
        assert window.wait() == 0
        window.record()
        clock.now += 1
    # The fourth waits until the first is 60 s old: sent at 1000, now 1003.
    assert window.wait() == pytest.approx(57)


def test_a_place_frees_when_the_oldest_is_60_s_old():
    clock = Clock()
    window = pacer.Window(2, clock)
    window.record()
    clock.now += 10
    window.record()
    clock.now = 1000 + 59.9
    assert window.wait() == pytest.approx(0.1)
    clock.now = 1000 + 60
    assert window.wait() == 0
    window.record()
    # The second, sent at 1010, is now the oldest.
    assert window.wait() == pytest.approx(10)


def test_no_window_holds_more_than_the_limit():
    """Requests as fast as the window lets them go, then every 60 s span counted."""
    clock = Clock()
    window = pacer.Window(9, clock)
    sent = []
    for _ in range(100):
        clock.now += window.wait()
        window.record()
        sent.append(clock.now)
        clock.now += 0.3
    assert max(sum(start <= t < start + 60 for t in sent) for start in sent) == 9


def test_a_limit_below_one_is_refused():
    with pytest.raises(ValueError):
        pacer.Window(0)


def test_the_hook_holds_concurrent_requests(monkeypatch):
    """Four requests at once through a limit of 2: two go, two are held until the
    first two are 60 s old, and each request's data comes back unchanged."""
    clock = Clock()
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(pacer.asyncio, "sleep", fake_sleep)
    hook = pacer.Pacer(2, margin=0.5)
    hook.window.clock = clock

    async def send_all():
        return await asyncio.gather(*(hook.async_pre_call_hook(None, None, {"n": n}, "anthropic_messages")
                                      for n in range(4)))

    results = asyncio.run(send_all())
    assert [data["n"] for data in results] == [0, 1, 2, 3]
    assert slept == [pytest.approx(60.5)]
    assert hook.held == 1
    assert len(hook.window.sent) == 2


def test_each_answer_s_prompt_size_is_logged(tmp_path, monkeypatch):
    """The service's counts from LiteLLM's logging payload, one line per answer;
    nothing without USAGE_LOG or without a payload."""
    hook = pacer.Pacer(9)
    log = tmp_path / "usage.jsonl"
    payload = {"model": "hosted_vllm/alias-glm-5.3-flash", "prompt_tokens": 41234,
               "completion_tokens": 310}

    asyncio.run(hook.async_log_success_event({"standard_logging_object": payload}, None, 0, 1))
    assert not log.exists()

    monkeypatch.setenv("USAGE_LOG", str(log))
    asyncio.run(hook.async_log_success_event({}, None, 0, 1))
    assert not log.exists()
    for _ in range(2):
        asyncio.run(hook.async_log_success_event({"standard_logging_object": payload}, None, 0, 1))
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(row["prompt_tokens"], row["completion_tokens"]) for row in rows] == [(41234, 310)] * 2
    assert rows[0]["model"] == "hosted_vllm/alias-glm-5.3-flash"
