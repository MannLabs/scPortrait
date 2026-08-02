from __future__ import annotations

import scportrait.data._datasets as datasets


def test_get_remote_dataset_redownloads_when_expected_file_missing(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    save_path = data_root / "example_dataset"
    save_path.mkdir(parents=True)

    captured = {}

    def _fake_download(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(datasets, "get_data_dir", lambda: data_root)
    monkeypatch.setattr(datasets, "_download", _fake_download)

    returned = datasets._get_remote_dataset(
        dataset="example_dataset",
        url="https://example.com/file.dat",
        name="file.dat",
        archive_format=None,
        outfile_name="file.dat",
    )

    assert returned == save_path / "file.dat"
    assert captured["output_path"] == str(save_path)
    assert captured["output_file_name"] == "file.dat"
    assert captured["overwrite"] is True


def test_get_remote_dataset_skips_download_when_expected_file_exists(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    save_path = data_root / "example_dataset"
    save_path.mkdir(parents=True)
    (save_path / "file.dat").write_text("ok")

    def _unexpected_download(**kwargs):
        raise AssertionError("_download should not be called when expected file already exists")

    monkeypatch.setattr(datasets, "get_data_dir", lambda: data_root)
    monkeypatch.setattr(datasets, "_download", _unexpected_download)

    returned = datasets._get_remote_dataset(
        dataset="example_dataset",
        url="https://example.com/file.dat",
        name="file.dat",
        archive_format=None,
        outfile_name="file.dat",
    )

    assert returned == save_path / "file.dat"


def test_get_remote_dataset_downloads_when_dataset_dir_missing(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    save_path = data_root / "example_dataset"

    captured = {}

    def _fake_download(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(datasets, "get_data_dir", lambda: data_root)
    monkeypatch.setattr(datasets, "_download", _fake_download)

    returned = datasets._get_remote_dataset(
        dataset="example_dataset",
        url="https://example.com/archive.zip",
        name=None,
        archive_format="zip",
        outfile_name=None,
    )

    assert returned == save_path
    assert captured["output_path"] == str(save_path)
    assert captured["overwrite"] is False


def test_get_remote_dataset_named_file_missing_dir_does_not_force_overwrite(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    save_path = data_root / "example_dataset"

    captured = {}

    def _fake_download(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(datasets, "get_data_dir", lambda: data_root)
    monkeypatch.setattr(datasets, "_download", _fake_download)

    returned = datasets._get_remote_dataset(
        dataset="example_dataset",
        url="https://example.com/file.dat",
        name="file.dat",
        archive_format=None,
        outfile_name="file.dat",
    )

    assert returned == save_path / "file.dat"
    assert captured["output_path"] == str(save_path)
    assert captured["output_file_name"] == "file.dat"
    assert captured["overwrite"] is False
