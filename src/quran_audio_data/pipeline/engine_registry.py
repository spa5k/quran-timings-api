from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

from .types import EngineAvailabilityPolicy, EngineOption, PipelineError


@runtime_checkable
class EngineProtocol(Protocol):
    def align(
        self, *, audio_wav_path: str, canonical_words: list, audio_duration_s: float, device: str
    ): ...


@dataclass(slots=True)
class RegisteredEngine:
    name: EngineOption
    instance: EngineProtocol
    available: bool
    reason: str | None = None


@dataclass(slots=True)
class EngineSelection:
    requested_engine: EngineOption
    engines_to_try: list[EngineOption]
    unavailable: dict[EngineOption, str]


class EngineRegistry:
    def __init__(
        self,
        *,
        nemo: EngineProtocol,
        whisperx: EngineProtocol,
        mfa: EngineProtocol,
    ) -> None:
        self._engines: dict[EngineOption, RegisteredEngine] = {
            "nemo": self._make_registered("nemo", nemo),
            "whisperx": self._make_registered("whisperx", whisperx),
            "mfa": self._make_registered("mfa", mfa),
        }

    def get(self, name: EngineOption) -> EngineProtocol:
        return self._engines[name].instance

    def availability(self) -> dict[EngineOption, RegisteredEngine]:
        return dict(self._engines)

    def select(
        self,
        *,
        requested_engine: EngineOption,
        multi_engine: list[EngineOption],
        policy: EngineAvailabilityPolicy,
    ) -> EngineSelection:
        ordered: list[EngineOption] = []
        for engine_name in multi_engine:
            if engine_name not in {"nemo", "whisperx", "mfa"}:
                continue
            if engine_name not in ordered:
                ordered.append(engine_name)
        for fallback in ("nemo", "whisperx", "mfa"):
            if fallback not in ordered:
                ordered.append(cast(EngineOption, fallback))

        unavailable: dict[EngineOption, str] = {}
        available: list[EngineOption] = []
        for engine_name in ordered:
            reg = self._engines[engine_name]
            if reg.available:
                available.append(engine_name)
            else:
                unavailable[engine_name] = reg.reason or "unavailable"

        if policy == "require_all" and unavailable:
            details = "; ".join(f"{name}={reason}" for name, reason in unavailable.items())
            raise PipelineError(f"engine_unavailable(require_all): {details}")

        if policy in {"require_all", "require_requested"} and requested_engine in unavailable:
            raise PipelineError(
                f"requested_engine_unavailable:{requested_engine}: {unavailable[requested_engine]}"
            )

        if not available:
            details = (
                "; ".join(f"{name}={reason}" for name, reason in unavailable.items())
                or "none available"
            )
            raise PipelineError(f"no_available_engines: {details}")

        selected_requested = requested_engine
        if selected_requested not in available:
            selected_requested = available[0]

        return EngineSelection(
            requested_engine=selected_requested,
            engines_to_try=available,
            unavailable=unavailable,
        )

    def _make_registered(self, name: EngineOption, engine: EngineProtocol) -> RegisteredEngine:
        available_fn = getattr(engine, "is_available", None)
        availability_error_fn = getattr(engine, "availability_error", None)
        if callable(available_fn):
            try:
                is_available = bool(available_fn())
            except Exception as exc:  # pragma: no cover
                return RegisteredEngine(
                    name=name, instance=engine, available=False, reason=str(exc)
                )
            if not is_available:
                reason = ""
                if callable(availability_error_fn):
                    try:
                        reason = str(availability_error_fn())
                    except Exception:
                        reason = ""
                return RegisteredEngine(
                    name=name,
                    instance=engine,
                    available=False,
                    reason=reason or "reported unavailable",
                )

        return RegisteredEngine(name=name, instance=engine, available=True, reason=None)
