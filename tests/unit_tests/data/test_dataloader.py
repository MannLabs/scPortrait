from scportrait.data._dataloader import _download


def test_download_existing_file_without_overwrite_skips_download(tmp_path, monkeypatch):
    output_dir = tmp_path / "dataset"
    output_dir.mkdir()
    existing_file = output_dir / "single_cells.h5sc"
    existing_file.write_bytes(b"existing-content")

    def _unexpected_get(*args, **kwargs):
        raise AssertionError("requests.get should not be called when file already exists and overwrite=False")

    monkeypatch.setattr("scportrait.data._dataloader.requests.get", _unexpected_get)

    _download(
        url="https://example.com/single_cells.h5sc",
        archive_format=None,
        output_file_name=existing_file.name,
        output_path=output_dir,
        overwrite=False,
    )

    assert existing_file.exists()
    assert existing_file.read_bytes() == b"existing-content"
