from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scportrait.pipeline._base import Logable, ProcessingStep


def _read_log_content(logable: Logable, tmp_path) -> str:
    log_path = tmp_path / logable.DEFAULT_LOG_NAME
    return log_path.read_text()


def _make_step(tmp_path, config: dict | str | None = None) -> ProcessingStep:
    if config is None:
        config = {"cache": str(tmp_path / "cache")}
    return ProcessingStep(config=config, directory=tmp_path / "step")


def test_logable_init_sets_debug_flag(tmp_path):
    """Logable should store the debug flag and normalize directory paths to str."""
    logable = Logable(directory=tmp_path, debug=False)
    assert logable.debug is False
    assert isinstance(logable.directory, str)


@pytest.mark.parametrize(
    ("message", "expected_lines"),
    [
        pytest.param("Testing", ["Testing"], id="single-line-string"),
        pytest.param("line1\nline2", ["line1", "line2"], id="multiline-string"),
        pytest.param({"alpha": 1, "beta": "x"}, ["alpha: 1", "beta: x"], id="dict-formatted"),
    ],
)
def test_logable_log_writes_expected_message_shapes(tmp_path, message, expected_lines):
    """Logable.log should persist strings, multiline strings, and dict payloads predictably."""
    logable = Logable(directory=tmp_path, debug=True)
    logable.log(message)

    log_content = _read_log_content(logable, tmp_path)
    for expected in expected_lines:
        assert expected in log_content


@pytest.mark.parametrize("config_source", ["dict", "path"])
def test_processing_step_init_normalizes_config_from_supported_sources(tmp_path, config_source):
    """ProcessingStep should accept both dict configs and config file paths."""
    config_dict = {"cache": str(tmp_path / "cache"), "setting1": "value1"}
    if config_source == "path":
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.safe_dump(config_dict))
        processing_step = ProcessingStep(config_path, tmp_path / "test_step", tmp_path, debug=True)
    else:
        processing_step = ProcessingStep(config_dict, tmp_path / "test_step", tmp_path, debug=True)

    assert processing_step.debug
    assert processing_step.config["setting1"] == "value1"


def test_processing_step_register_parameter(tmp_path):
    """register_parameter should add missing keys to the step config."""
    config = {"setting1": "value1"}
    processing_step = ProcessingStep(config, tmp_path / "test_step", tmp_path)

    processing_step.register_parameter("setting2", "value2")
    assert "setting2" in processing_step.config
    assert "value2" == processing_step.config["setting2"]


def test_processing_step_get_directory(tmp_path):
    """get_directory should return the configured step directory as a string path."""
    config = {"setting1": "value1"}
    processing_step = ProcessingStep(config, tmp_path / "test_step", tmp_path)
    assert str(tmp_path / "test_step") == processing_step.get_directory()


def test_processing_step_create_and_clear_temp_dir_in_cache(tmp_path):
    """create_temp_dir/clear_temp_dir should manage a temporary workspace inside cache."""
    cache_dir = tmp_path / "cache"
    step = _make_step(tmp_path, {"cache": str(cache_dir)})

    step.create_temp_dir()
    tmp_dir_path = step._tmp_dir_path
    assert step._tmp_dir_path.startswith(str(cache_dir))
    assert Path(tmp_dir_path).is_dir()

    step.clear_temp_dir()
    assert not Path(tmp_dir_path).exists()


@pytest.mark.parametrize(
    ("invoker", "warning_text"),
    [
        pytest.param(lambda step: step(), "No process method defined.", id="missing-process"),
        pytest.param(lambda step: step.__call_empty__(), "No return_empty_mask method defined.", id="missing-empty"),
    ],
)
def test_processing_step_emits_warnings_when_required_methods_are_missing(tmp_path, invoker, warning_text):
    """Calling step entrypoints without required subclass methods should warn and return None."""
    step = _make_step(tmp_path)
    with pytest.warns(UserWarning, match=warning_text):
        result = invoker(step)
    assert result is None


def test_processing_step_call_empty_uses_temp_dir_lifecycle_for_callable(tmp_path):
    """__call_empty__ should create and cleanup a temporary workspace when method exists."""

    class EmptyStep(ProcessingStep):
        def return_empty_mask(self):
            return {"tmp_dir_exists": Path(self._tmp_dir_path).is_dir()}

    step = EmptyStep(config={"cache": str(tmp_path / "cache")}, directory=tmp_path / "step")
    result = step.__call_empty__()

    assert result == {"tmp_dir_exists": True}
    assert not hasattr(step, "_tmp_dir_path")


def test_processing_step_register_parameter_nested_keys_not_supported(tmp_path):
    """register_parameter should reject nested list keys until that API is implemented."""
    step = _make_step(tmp_path)

    with pytest.raises(NotImplementedError):
        step.register_parameter(["a", "b"], 1)
