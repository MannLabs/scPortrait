from __future__ import annotations

import io
import zipfile

import pytest

import scportrait.data._dataloader as dataloader


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self.headers = {"content-length": str(len(payload))}

    def raise_for_status(self):
        return None

    def iter_content(self, block_size: int):
        for start in range(0, len(self._payload), block_size):
            yield self._payload[start : start + block_size]


class _HttpErrorResponse:
    headers = {"content-length": "0"}

    def raise_for_status(self):
        raise dataloader.requests.HTTPError("boom")

    def iter_content(self, block_size: int):
        yield b""


class _InterruptedStreamResponse:
    headers = {"content-length": "12"}

    def raise_for_status(self):
        return None

    def iter_content(self, block_size: int):
        yield b"abc"
        raise RuntimeError("stream interrupted")


@pytest.mark.parametrize(
    ("overwrite", "expected_content", "expect_network_call"),
    [
        pytest.param(False, b"existing-content", False, id="existing-file-no-overwrite"),
        pytest.param(True, b"new-content", True, id="existing-file-overwrite"),
    ],
)
def test_download_existing_file_overwrite_behavior(
    tmp_path, monkeypatch, overwrite, expected_content, expect_network_call
):
    """When the target exists, overwrite=False keeps data and overwrite=True fetches and replaces it."""
    output_dir = tmp_path / "dataset"
    output_dir.mkdir()
    output_file = output_dir / "single_cells.h5sc"
    output_file.write_bytes(b"existing-content")

    if expect_network_call:
        monkeypatch.setattr(dataloader.requests, "get", lambda *args, **kwargs: _FakeResponse(b"new-content"))
    else:
        monkeypatch.setattr(
            dataloader.requests,
            "get",
            lambda *args, **kwargs: pytest.fail("requests.get should not be called when overwrite=False"),
        )

    dataloader._download(
        url="https://example.com/single_cells.h5sc",
        archive_format=None,
        output_file_name=output_file.name,
        output_path=output_dir,
        overwrite=overwrite,
    )

    assert output_file.read_bytes() == expected_content


def test_download_writes_expected_file_contents(tmp_path, monkeypatch):
    """Successful non-archive downloads should write bytes and finalize by removing the .part file."""
    output_dir = tmp_path / "download"
    payload = b"abcdef123456"
    output_file = output_dir / "payload.bin"

    def _fake_get(url, stream, headers):
        assert url == "https://example.com/payload.bin"
        assert stream is True
        assert headers == {"User-Agent": "scPortrait"}
        return _FakeResponse(payload)

    monkeypatch.setattr(dataloader.requests, "get", _fake_get)

    dataloader._download(
        url="https://example.com/payload.bin",
        archive_format=None,
        output_file_name=output_file.name,
        output_path=output_dir,
        block_size=4,
        overwrite=False,
    )

    assert output_file.exists()
    assert output_file.read_bytes() == payload
    assert not (output_dir / f"{output_file.name}.part").exists()


def test_download_zip_archive_extracts_and_removes_archive_file(tmp_path, monkeypatch):
    """Archive downloads should unpack into output_path and remove the downloaded archive file."""
    output_dir = tmp_path / "archive"
    archive_name = "bundle.zip"
    archive_path = output_dir / archive_name

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("inner/data.txt", "hello-archive")
    payload = archive_buffer.getvalue()

    monkeypatch.setattr(dataloader.requests, "get", lambda *args, **kwargs: _FakeResponse(payload))

    dataloader._download(
        url="https://example.com/bundle.zip",
        archive_format="zip",
        output_file_name=archive_name,
        output_path=output_dir,
    )

    assert (output_dir / "inner" / "data.txt").read_text() == "hello-archive"
    assert not archive_path.exists()


@pytest.mark.parametrize(
    ("response", "expected_exception", "part_should_exist"),
    [
        pytest.param(_HttpErrorResponse(), dataloader.requests.HTTPError, False, id="http-error-before-write"),
        pytest.param(_InterruptedStreamResponse(), RuntimeError, True, id="stream-error-during-write"),
    ],
)
def test_download_failure_modes(tmp_path, monkeypatch, response, expected_exception, part_should_exist):
    """Download failures should raise and expose current temporary-file behavior for each failure stage."""
    output_dir = tmp_path / "errors"
    output_file = output_dir / "payload.bin"
    part_file = output_dir / "payload.bin.part"

    monkeypatch.setattr(dataloader.requests, "get", lambda *args, **kwargs: response)

    with pytest.raises(expected_exception):
        dataloader._download(
            url="https://example.com/payload.bin",
            archive_format=None,
            output_file_name=output_file.name,
            output_path=output_dir,
        )

    assert not output_file.exists()
    assert part_file.exists() is part_should_exist


def test_download_with_default_name_and_path(tmp_path, monkeypatch):
    """When name/path are omitted, download should use tempdir and generated scportrait_tmp_* filename."""
    payload = b"default-path-payload"

    monkeypatch.setattr(dataloader.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(dataloader, "choice", lambda _: "x")
    monkeypatch.setattr(dataloader.requests, "get", lambda *args, **kwargs: _FakeResponse(payload))

    dataloader._download(
        url="https://example.com/default.bin",
        archive_format=None,
        output_file_name=None,
        output_path=None,
    )

    default_file = tmp_path / "scportrait_tmp_xxxxxxxxxx"
    assert default_file.exists()
    assert default_file.read_bytes() == payload
