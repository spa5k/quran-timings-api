from __future__ import annotations

import atexit
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from quran_audio_data.core.settings import get_settings


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class RetryableStatusError(RuntimeError):
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(f"retryable status code: {response.status_code}")


_client = httpx.Client(follow_redirects=True)
atexit.register(_client.close)


def _build_retry(retries: int, initial: float, maximum: float, jitter: float):
    return retry(
        reraise=True,
        stop=stop_after_attempt(max(1, retries + 1)),
        wait=wait_exponential_jitter(
            initial=initial,
            max=maximum,
            jitter=jitter,
        ),
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.NetworkError, RetryableStatusError)
        ),
    )


def get_json_with_retry(
    *,
    url: str,
    params: dict[str, str] | None = None,
    timeout_s: float | None = None,
    retries: int | None = None,
    retry_backoff_s: float | None = None,
    retry_max_backoff_s: float | None = None,
    retry_jitter_s: float | None = None,
) -> Any:
    settings = get_settings()
    timeout = timeout_s if timeout_s is not None else settings.request_timeout_s
    attempts = retries if retries is not None else settings.request_retries
    initial = retry_backoff_s if retry_backoff_s is not None else settings.retry_backoff_s
    maximum = (
        retry_max_backoff_s
        if retry_max_backoff_s is not None
        else settings.retry_max_backoff_s
    )
    jitter = retry_jitter_s if retry_jitter_s is not None else settings.retry_jitter_s

    @_build_retry(attempts, initial, maximum, jitter)
    def _run() -> httpx.Response:
        response = _client.get(url, params=params, timeout=timeout)
        if response.status_code in RETRYABLE_STATUS_CODES:
            raise RetryableStatusError(response)
        response.raise_for_status()
        return response

    response = _run()
    return response.json()


def get_bytes_with_retry(
    *,
    url: str,
    params: dict[str, str] | None = None,
    timeout_s: float | None = None,
    retries: int | None = None,
    retry_backoff_s: float | None = None,
    retry_max_backoff_s: float | None = None,
    retry_jitter_s: float | None = None,
) -> bytes:
    settings = get_settings()
    timeout = timeout_s if timeout_s is not None else settings.request_timeout_s
    attempts = retries if retries is not None else settings.request_retries
    initial = retry_backoff_s if retry_backoff_s is not None else settings.retry_backoff_s
    maximum = (
        retry_max_backoff_s
        if retry_max_backoff_s is not None
        else settings.retry_max_backoff_s
    )
    jitter = retry_jitter_s if retry_jitter_s is not None else settings.retry_jitter_s

    @_build_retry(attempts, initial, maximum, jitter)
    def _run() -> httpx.Response:
        response = _client.get(url, params=params, timeout=timeout)
        if response.status_code in RETRYABLE_STATUS_CODES:
            raise RetryableStatusError(response)
        response.raise_for_status()
        return response

    response = _run()
    return response.content


def get_json_or_none(
    *,
    url: str,
    params: dict[str, str] | None = None,
    timeout_s: float | None = None,
    retries: int | None = None,
    retry_backoff_s: float | None = None,
) -> Any | None:
    try:
        return get_json_with_retry(
            url=url,
            params=params,
            timeout_s=timeout_s,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
        )
    except Exception:
        return None
