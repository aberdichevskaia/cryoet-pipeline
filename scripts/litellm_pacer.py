"""Keep a LiteLLM proxy under a service's requests-per-minute limit.

scripts/open_model_cases.sbatch copies this file next to the LiteLLM config it
writes and names `litellm_pacer.pacer` under litellm_settings.callbacks; LiteLLM
imports a callback module from the config's folder (get_instance_fn in
proxy/types_utils/utils.py, 1.106.5, read 11 November 2026) and runs each
callback's async_pre_call_hook before every request, /v1/messages included
(common_request_processing.py).

Why: Blablador allows users outside FZJ 10 requests per 60 s on most models,
and a user over the limit is throttled for 60 s, doubling up to 3600 s
(sdlaml.pages.jsc.fz-juelich.de/ai/guides/blablador_api_access, read 11 November
2026). Claude Code sends one request per turn, and Claude made about 16 turns a
minute in the evaluation cases, so an unpaced run can be locked out for an hour,
longer than a turn may take. The hook holds a request until fewer than
REQUESTS_PER_MINUTE (9) were sent in the last 60 s. LiteLLM's own retries skip
the hook, so the job sets them to 0 and leaves retrying to the client, whose
every attempt comes through here.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import deque
from collections.abc import Callable

WINDOW = 60.0


class Window:
    """When the next request may go so that no 60 s holds more than `per_minute`."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic):
        if per_minute < 1:
            raise ValueError(f"per_minute must be at least 1, not {per_minute}")
        self.per_minute = per_minute
        self.clock = clock
        self.sent: deque[float] = deque()

    def wait(self) -> float:
        """Seconds until a request may go; 0 when it may go now."""
        now = self.clock()
        while self.sent and now - self.sent[0] >= WINDOW:
            self.sent.popleft()
        if len(self.sent) < self.per_minute:
            return 0.0
        return WINDOW - (now - self.sent[0])

    def record(self) -> None:
        self.sent.append(self.clock())


try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # the tests read Window without LiteLLM installed
    CustomLogger = object  # type: ignore[assignment,misc]


class Pacer(CustomLogger):  # type: ignore[misc,valid-type]
    def __init__(self, per_minute: int, margin: float = 0.5):
        super().__init__()
        self.window = Window(per_minute)
        self.margin = margin
        self.lock = asyncio.Lock()
        self.held = 0

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        # One at a time, so that two requests never take the same free place.
        async with self.lock:
            while (delay := self.window.wait()) > 0:
                self.held += 1
                print(f"litellm_pacer: holding a request {delay + self.margin:.1f} s "
                      f"({self.window.per_minute} per {WINDOW:.0f} s; {self.held} held so far)",
                      file=sys.stderr, flush=True)
                await asyncio.sleep(delay + self.margin)
            self.window.record()
        return data

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Each answer's prompt size, to USAGE_LOG as a JSON line. LiteLLM's translated
        stream opens with message_start before the service has answered, so its
        input_tokens are 0 (_create_initial_usage_delta in
        llms/anthropic/pass_through/adapters/streaming_iterator.py, 1.106.5), and the
        client's transcript records 0: run_cases.py's largest_context stays empty.
        The service's own count reaches LiteLLM's logging payload."""
        path = os.environ.get("USAGE_LOG")
        payload = kwargs.get("standard_logging_object") or {}
        if not path or not payload:
            return
        row = {"time": time.time(), "model": payload.get("model"),
               "prompt_tokens": payload.get("prompt_tokens"),
               "completion_tokens": payload.get("completion_tokens")}
        with open(path, "a") as log:
            log.write(json.dumps(row) + "\n")


pacer = Pacer(int(os.environ.get("REQUESTS_PER_MINUTE", "9")))
