from __future__ import annotations

import quran_audio_data.alignment.mfa_aligner as mfa_aligner
from quran_audio_data.alignment.mfa_aligner import MFAAligner, _extract_mfa_words


def test_mfa_aligner_prefers_command_template(monkeypatch) -> None:
    monkeypatch.setattr(mfa_aligner.shutil, "which", lambda _name: None)
    aligner = MFAAligner(command_template="echo ok")
    assert aligner.is_available()
    assert aligner._mode == "template"


def test_mfa_aligner_uses_local_when_probe_succeeds(monkeypatch) -> None:
    def fake_which(name: str):
        if name == "mfa":
            return "/bin/mfa"
        return None

    monkeypatch.setattr(mfa_aligner.shutil, "which", fake_which)
    monkeypatch.setattr(mfa_aligner, "_probe_local_mfa", lambda _path: True)

    aligner = MFAAligner()
    assert aligner.is_available()
    assert aligner._mode == "local"


def test_mfa_aligner_uses_docker_when_local_is_broken(monkeypatch) -> None:
    def fake_which(name: str):
        if name == "mfa":
            return "/bin/mfa"
        if name == "docker":
            return "/bin/docker"
        return None

    monkeypatch.setattr(mfa_aligner.shutil, "which", fake_which)
    monkeypatch.setattr(mfa_aligner, "_probe_local_mfa", lambda _path: False)

    aligner = MFAAligner()
    assert aligner.is_available()
    assert aligner._mode == "docker"


def test_mfa_aligner_reports_unavailable_when_no_runtime(monkeypatch) -> None:
    monkeypatch.setattr(mfa_aligner.shutil, "which", lambda _name: None)

    aligner = MFAAligner()
    assert not aligner.is_available()
    assert aligner._mode == "unavailable"
    assert "Neither" in aligner.availability_error()


def test_extract_mfa_words_supports_tiers_entries_shape() -> None:
    payload = {
        "tiers": {
            "words": {
                "entries": [
                    [0.0, 0.2, "foo"],
                    [0.2, 0.4, "bar"],
                ]
            }
        }
    }
    words = _extract_mfa_words(payload)
    assert words == [("foo", 0.0, 0.2), ("bar", 0.2, 0.4)]
