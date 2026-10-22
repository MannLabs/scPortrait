from __future__ import annotations

import pytest

import scportrait.data._datasets as datasets


@pytest.mark.parametrize(
    ("directory_exists", "file_exists", "name", "archive_format", "outfile_name", "overwrite"),
    [
        pytest.param(True, False, "file.dat", None, "file.dat", True, id="missing-file-in-existing-dir"),
        pytest.param(True, True, "file.dat", None, "file.dat", False, id="cached-file"),
        pytest.param(False, False, None, "zip", None, False, id="missing-dir-archive"),
        pytest.param(False, False, "file.dat", None, "file.dat", False, id="missing-dir-named-file"),
    ],
)
def test_get_remote_dataset_cache_and_download_policy(
    tmp_path,
    monkeypatch,
    directory_exists,
    file_exists,
    name,
    archive_format,
    outfile_name,
    overwrite,
):
    data_root = tmp_path / "data"
    save_path = data_root / "example_dataset"
    if directory_exists:
        save_path.mkdir(parents=True)
    if file_exists:
        (save_path / outfile_name).write_text("cached")

    calls = []

    def _fake_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(datasets, "get_data_dir", lambda: data_root)
    monkeypatch.setattr(datasets, "_download", _fake_download)

    returned = datasets._get_remote_dataset(
        dataset="example_dataset",
        url="https://example.com/download",
        name=name,
        archive_format=archive_format,
        outfile_name=outfile_name,
    )

    expected_path = save_path / outfile_name if outfile_name else save_path
    assert returned == expected_path
    if file_exists:
        assert calls == []
        return

    assert len(calls) == 1
    assert calls[0]["output_path"] == str(save_path)
    assert calls[0]["output_file_name"] == outfile_name
    assert calls[0]["overwrite"] is overwrite
