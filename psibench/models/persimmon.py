"""Persimmon (humans& user model) as a patient simulator, ported from persimmon_psibench.ipynb.

Exposes `PersimmonPatient`, a drop-in for `PatientAgent` (same `respond(history, therapist_message)`),
so `run_session` in generate_conversations.py stays a plain synchronous loop. Each session owns one API
conversation and a private event loop, which makes it safe to run sessions in a ThreadPoolExecutor.
"""

import asyncio
import re
import threading
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

THERAPIST_NAME = "Therapist"
DEFAULT_MODEL = "persimmon-v0.1"
DEFAULT_REQUESTS_PER_MINUTE = 50   # stays under the key's 60/min bucket

T = TypeVar("T")


class PersimmonRefused(Exception):
    """The API's content moderation refused this profile (e.g. violence or self-harm)."""


def render_value(value: object) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{key}: {render_value(inner)}" for key, inner in value.items())
    if isinstance(value, list):
        return ", ".join(render_value(item) for item in value)
    return str(value)


def patient_name(profile: Dict[str, Any]) -> str:
    name = profile.get("name")
    if isinstance(name, str) and name.strip() and name.strip().upper() != "N/A":
        return name.strip()
    return "Client"


def profile_description(profile: Dict[str, Any]) -> str:
    lines = ["A client in a counselling conversation about their mental health."]
    lines += [f"{key.capitalize()}: {render_value(value)}" for key, value in profile.items() if key != "name"]
    return "\n".join(lines)


def session_setting(name: str) -> str:
    return (f"A one-on-one counselling session. {THERAPIST_NAME} is a professional therapist; "
            f"{name} is the client who came for support. They talk in turns.")


# Request pacing is shared by every session thread in the process.
_pace_lock = threading.Lock()
_next_request_at = 0.0


def _reserve_request_slot(requests_per_minute: float) -> float:
    """Return how long to wait before this request may start, spacing starts evenly."""
    global _next_request_at
    with _pace_lock:
        now = time.monotonic()
        start = max(now, _next_request_at)
        _next_request_at = start + 60.0 / requests_per_minute
        return start - now


class PersimmonPatient:
    """Patient simulator backed by the Persimmon API. Call `close()` when the session is done."""

    def __init__(self, patient_profile: Dict[str, Any], config: Dict[str, Any], session_id: Any = None, source: Any = None):
        # Imported lazily so the other simulators don't need the SDK installed.
        from humansand import AsyncPersimmon, ContentPolicyError, RateLimitError

        self._ContentPolicyError = ContentPolicyError
        self._RateLimitError = RateLimitError

        settings = config.get("persimmon") or {}
        self.model: str = settings.get("model", DEFAULT_MODEL)
        self.harness: bool = settings.get("harness", True)            # the API's default quality checks
        self.max_tokens: Optional[int] = settings.get("max_tokens")   # None = API default of 1,024 (max 4,096)
        self.requests_per_minute: float = settings.get("requests_per_minute", DEFAULT_REQUESTS_PER_MINUTE)
        self.max_total_wait: float = settings.get("max_total_wait_seconds", 24 * 3600)

        self.profile = patient_profile
        self.name = patient_name(patient_profile)
        self.setting = session_setting(self.name)
        self.session_id = session_id
        self.source = source
        self.turns: List[Dict[str, Any]] = []
        self.conversation = None

        self._loop = asyncio.new_event_loop()
        self._client = AsyncPersimmon()   # reads USER_MODEL_API_KEY

    def _run(self, coro: Awaitable[T]) -> T:
        return self._loop.run_until_complete(coro)

    async def _call(self, call: Callable[[], Awaitable[T]]) -> T:
        """Await `call()`, pacing requests and waiting out RateLimitErrors. The SDK already retries short
        waits; what reaches us is mostly the daily token budget ("retry after 3600s")."""
        waited = 0.0
        while True:
            delay = _reserve_request_slot(self.requests_per_minute)
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                return await call()
            except self._RateLimitError as error:
                match = re.search(r"retry after (\d+(?:\.\d+)?)s", str(error))
                wait = min(float(match.group(1)) if match else 60.0, 600.0)
                if waited + wait > self.max_total_wait:
                    raise
                waited += wait
                await asyncio.sleep(wait)

    async def _open(self) -> None:
        try:
            self.conversation = await self._call(lambda: self._client.conversations.create(
                model=self.model,
                participants=[
                    {"name": self.name, "description": profile_description(self.profile)},
                    {"name": THERAPIST_NAME, "description": "A professional therapist.", "generate_content": False},
                ],
                setting=self.setting,
                metadata={"benchmark": "psibench", "session_id": str(self.session_id), "source": str(self.source).lower()},
            ))
        except self._ContentPolicyError as error:
            raise PersimmonRefused(str(error)) from error

    async def _turn(self, therapist_message: str) -> str:
        if self.conversation is None:
            await self._open()
        self.conversation.append_turn({"name": THERAPIST_NAME, "content": therapist_message})
        try:
            turn = await self._call(lambda: self.conversation.get_turn(
                speaker=self.name, harness=self.harness, max_tokens=self.max_tokens))
        except self._ContentPolicyError as error:
            raise PersimmonRefused(str(error)) from error
        if not turn.content or not turn.content.strip():
            raise ValueError(f"Persimmon returned no content for {self.name} (finish_reason={turn.finish_reason})")
        self.conversation.append_turn(turn)
        self.turns.append({"turn_id": turn.id, "finish_reason": turn.finish_reason,
                           "prompt_tokens": turn.usage.prompt_tokens, "completion_tokens": turn.usage.completion_tokens})
        return turn.content

    def respond(self, conversation_history: List[Dict[str, str]], therapist_message: str) -> str:
        """Same contract as PatientAgent.respond. The API conversation is stateful on our side, so only the
        new therapist message is appended; `conversation_history` is already in it."""
        return self._run(self._turn(therapist_message))

    def audit(self) -> Dict[str, Any]:
        """Per-session record stored under `persimmon` in the session JSON."""
        return {
            "model": self.model, "harness": self.harness, "max_tokens": self.max_tokens,
            "patient_name": self.name, "setting": self.setting,
            "conversation_id": self.conversation.id if self.conversation is not None else None,
            "turns": self.turns,
        }

    def close(self) -> None:
        """Archive the API conversation (housekeeping only) and release the client and event loop."""
        try:
            if self.conversation is not None:
                try:
                    self._run(self._client.conversations.archive(self.conversation.id))
                except Exception as error:
                    print(f"archive skipped for {self.conversation.id}: {error}")
            # The notebook uses the client as `async with`, so __aexit__ is the documented way to release it.
            self._run(self._client.__aexit__(None, None, None))
        except Exception:
            pass
        finally:
            self._loop.close()
