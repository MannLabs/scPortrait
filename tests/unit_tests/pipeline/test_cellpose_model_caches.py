from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError

import pytest

from scportrait.pipeline.segmentation.workflows import _model_caches as model_caches


def _http_error() -> HTTPError:
    return HTTPError(url="https://example.invalid/model", code=503, msg="service down", hdrs=None, fp=None)


def test_scportrait_cache_model_path_uses_cached_file_without_redownloading(tmp_path, monkeypatch):
    model_dir = tmp_path / "cellpose_models"
    model_dir.mkdir(parents=True)
    cached_model = model_dir / "nuclei"
    cached_model.write_text("cached model", encoding="utf-8")

    download_calls: list[dict[str, object]] = []

    def _fake_download(url: str, destination: str, progress: bool):
        download_calls.append({"url": url, "destination": destination, "progress": progress})

    monkeypatch.setattr(model_caches, "_get_model_dir", lambda: model_dir)
    monkeypatch.setattr(model_caches.utils, "download_url_to_file", _fake_download)

    resolved = model_caches._scportrait_cache_model_path("nuclei")

    assert resolved == str(cached_model)
    assert download_calls == []


def test_scportrait_cache_model_path_downloads_missing_file_once(tmp_path, monkeypatch):
    model_dir = tmp_path / "cellpose_models"
    model_dir.mkdir(parents=True)
    expected_path = model_dir / "cpsam"

    download_calls: list[dict[str, object]] = []

    def _fake_download(url: str, destination: str, progress: bool):
        download_calls.append({"url": url, "destination": destination, "progress": progress})
        Path(destination).write_text("downloaded model", encoding="utf-8")

    monkeypatch.setattr(model_caches, "_get_model_dir", lambda: model_dir)
    monkeypatch.setattr(model_caches.utils, "download_url_to_file", _fake_download)

    resolved = model_caches._scportrait_cache_model_path("cpsam")

    assert resolved == str(expected_path)
    assert expected_path.exists()
    assert download_calls == [
        {
            "url": model_caches._make_zenodo_download_link(model_caches.ZENODO_RECORD_ID, "cpsam"),
            "destination": str(expected_path),
            "progress": True,
        }
    ]


def test_download_model_cp4_name_uses_cellpose_model_path_and_returns_string(monkeypatch):
    expected = Path("/cellpose/cpsam")
    monkeypatch.setattr(model_caches.models, "model_path", lambda name: Path(f"/cellpose/{name}"), raising=False)
    monkeypatch.setattr(
        model_caches,
        "_model_path",
        lambda *_args, **_kwargs: pytest.fail("_model_path fallback must not be used when model_path succeeds"),
    )

    resolved = model_caches._download_model("cpsam")

    assert Path(resolved) == expected
    assert isinstance(resolved, str)


def test_download_model_falls_back_to_scportrait_cache_when_cellpose_lookup_fails(monkeypatch):
    monkeypatch.setattr(
        model_caches.models,
        "model_path",
        lambda _name: (_ for _ in ()).throw(FileNotFoundError("missing in default cache")),
        raising=False,
    )

    model_calls: list[str] = []
    size_calls: list[str] = []
    monkeypatch.setattr(model_caches, "_model_path", lambda name: (model_calls.append(name), "/backup/cpsam")[1])
    monkeypatch.setattr(model_caches, "_size_model_path", lambda name: size_calls.append(name))

    resolved = model_caches._download_model("cpsam")

    assert resolved == "/backup/cpsam"
    assert model_calls == ["cpsam"]
    assert size_calls == ["cpsam"]


def test_download_model_raises_actionable_error_when_resolution_fails(monkeypatch):
    monkeypatch.setattr(
        model_caches.models,
        "model_path",
        lambda _name: (_ for _ in ()).throw(_http_error()),
        raising=False,
    )
    monkeypatch.setattr(
        model_caches,
        "_model_path",
        lambda _name: (_ for _ in ()).throw(_http_error()),
    )

    with pytest.raises(FileNotFoundError, match=r"Could not resolve Cellpose model 'cpsam'"):
        model_caches._download_model("cpsam")


def test_download_model_legacy_name_raises_actionable_error_when_cache_download_fails(monkeypatch):
    monkeypatch.setattr(
        model_caches,
        "_model_path",
        lambda _name: (_ for _ in ()).throw(_http_error()),
    )
    monkeypatch.setattr(
        model_caches,
        "_size_model_path",
        lambda *_args, **_kwargs: pytest.fail("_size_model_path should not run after model download failure"),
    )

    with pytest.raises(FileNotFoundError, match=r"Could not resolve Cellpose model 'nuclei'"):
        model_caches._download_model("nuclei")
