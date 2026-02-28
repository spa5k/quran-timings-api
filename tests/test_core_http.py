from __future__ import annotations

import httpx

from quran_audio_data.core import http as core_http


class _FakeResponse:
    def __init__(self, *, status_code: int, payload: dict | None = None, content: bytes = b"") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.content = content
        self.request = httpx.Request("GET", "https://example.test")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response = httpx.Response(self.status_code, request=self.request)
            raise httpx.HTTPStatusError("HTTP failure", request=self.request, response=response)

    def json(self):
        return self._payload


def test_get_json_with_retry_retries_retryable_status(monkeypatch) -> None:
    calls = {"count": 0}

    class _FakeClient:
        def get(self, url, params=None, timeout=None):  # noqa: ANN001
            calls["count"] += 1
            if calls["count"] < 3:
                return _FakeResponse(status_code=503)
            return _FakeResponse(status_code=200, payload={"ok": True})

    monkeypatch.setattr(core_http, "_client", _FakeClient())

    payload = core_http.get_json_with_retry(
        url="https://example.test/retry",
        timeout_s=0.01,
        retries=2,
        retry_backoff_s=0.0,
        retry_max_backoff_s=0.0,
        retry_jitter_s=0.0,
    )

    assert calls["count"] == 3
    assert payload == {"ok": True}


def test_get_json_or_none_returns_none_after_timeout_retries(monkeypatch) -> None:
    calls = {"count": 0}

    class _FakeClient:
        def get(self, url, params=None, timeout=None):  # noqa: ANN001
            calls["count"] += 1
            request = httpx.Request("GET", url)
            raise httpx.ReadTimeout("timed out", request=request)

    monkeypatch.setattr(core_http, "_client", _FakeClient())

    payload = core_http.get_json_or_none(
        url="https://example.test/timeout",
        timeout_s=0.01,
        retries=1,
        retry_backoff_s=0.0,
    )

    assert payload is None
    assert calls["count"] == 2
