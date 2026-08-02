import importlib
import sys

import numpy as np
import pytest
from skimage import data

from scportrait._utils.optional_dependencies import import_optional_dependency


def test_import_optional_dependency_returns_attribute():
    sqrt = import_optional_dependency("math", attribute="sqrt")
    assert sqrt(16) == 4


def test_import_optional_dependency_returns_none_when_non_raising():
    assert import_optional_dependency("scportrait_missing_test_dependency", raise_on_missing=False) is None


def test_import_optional_dependency_missing_raises_guided_error():
    with pytest.raises(ImportError, match=r"scportrait\[plotting\]"):
        import_optional_dependency(
            "scportrait_missing_test_dependency",
            package_name="fake-package",
            feature="the plotting capabilities",
            install_hint="pip install 'scportrait[plotting]'",
        )


def test_import_optional_dependency_reraises_transitive_module_not_found(monkeypatch):
    original = ModuleNotFoundError("No module named 'transitive_dependency'")
    original.name = "transitive_dependency"

    def _raise_transitive_failure(_module_name):
        raise original

    monkeypatch.setattr(importlib, "import_module", _raise_transitive_failure)

    with pytest.raises(ModuleNotFoundError, match="transitive_dependency") as exc_info:
        import_optional_dependency("optional_package")

    assert exc_info.value is original


def test_import_optional_dependency_non_raising_still_reraises_transitive_module_not_found(monkeypatch):
    original = ModuleNotFoundError("No module named 'transitive_dependency'")
    original.name = "transitive_dependency"

    def _raise_transitive_failure(_module_name):
        raise original

    monkeypatch.setattr(importlib, "import_module", _raise_transitive_failure)

    with pytest.raises(ModuleNotFoundError, match="transitive_dependency") as exc_info:
        import_optional_dependency("optional_package", raise_on_missing=False)

    assert exc_info.value is original


def test_import_optional_dependency_reraises_plain_import_error(monkeypatch):
    original = ImportError("cannot import name 'foo' from 'bar'")

    def _raise_import_error(_module_name):
        raise original

    monkeypatch.setattr(importlib, "import_module", _raise_import_error)

    with pytest.raises(ImportError, match="cannot import name") as exc_info:
        import_optional_dependency("optional_package")

    assert exc_info.value is original


def test_deprecation_helper_exports_decorator():
    from scportrait._utils.deprecation import deprecated

    assert callable(deprecated)


def test_check_for_spatialdata_plot_missing(monkeypatch):
    from scportrait.pipeline._utils.helper import _check_for_spatialdata_plot

    monkeypatch.setitem(sys.modules, "spatialdata_plot", None)

    with pytest.raises(ImportError, match="Extended plotting capabilities"):
        _check_for_spatialdata_plot()


def test_plotting_utils_import_without_matplotlib_scalebar_keeps_custom_cmap_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib_scalebar", None)
    monkeypatch.setitem(sys.modules, "matplotlib_scalebar.scalebar", None)
    sys.modules.pop("scportrait.plotting._utils", None)

    plotting_utils = importlib.import_module("scportrait.plotting._utils")

    cmap, norm = plotting_utils._custom_cmap()

    assert cmap.N == 3
    assert norm is not None


def test_add_scalebar_missing_matplotlib_scalebar_raises_guided_error(monkeypatch):
    from scportrait.plotting._utils import add_scalebar

    monkeypatch.setitem(sys.modules, "matplotlib_scalebar", None)
    monkeypatch.setitem(sys.modules, "matplotlib_scalebar.scalebar", None)

    class _DummyAxes:
        def add_artist(self, _artist):
            raise AssertionError("add_artist should not be called when matplotlib_scalebar is missing")

    with pytest.raises(ImportError, match=r"scportrait\[plotting\]"):
        add_scalebar(_DummyAxes(), resolution=1.0)


def test_project_view_sdata_missing_napari_spatialdata_raises_guided_error(monkeypatch):
    from scportrait.pipeline.project import Project

    monkeypatch.setitem(sys.modules, "napari_spatialdata", None)

    project = Project.__new__(Project)
    project.log = lambda *_args, **_kwargs: None

    with pytest.raises(ImportError, match=r"scportrait\[plotting\]"):
        Project.view_sdata(project)


def test_import_transformers_missing_raises_guided_error(monkeypatch):
    from scportrait.pipeline.featurization import _import_transformers

    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(ImportError, match=r"scportrait\[convnext\]"):
        _import_transformers()


def test_zstack_compression_module_import_without_mahotas(monkeypatch):
    import scportrait.processing.images._zstack_compression as zstack_compression

    monkeypatch.setitem(sys.modules, "mahotas", None)

    reloaded = importlib.reload(zstack_compression)

    assert hasattr(reloaded, "EDF")


def test_edf_missing_mahotas_raises_guided_error(monkeypatch):
    from scportrait.processing.images._zstack_compression import EDF

    monkeypatch.setitem(sys.modules, "mahotas", None)

    with pytest.raises(ImportError, match=r"scportrait\[zstack\]"):
        EDF(np.zeros((2, 4, 4), dtype=np.uint16))


def test_segmentation_utils_module_import_without_skfmm(monkeypatch):
    import scportrait.pipeline._utils.segmentation as segmentation_utils

    monkeypatch.setitem(sys.modules, "skfmm", None)

    reloaded = importlib.reload(segmentation_utils)

    assert hasattr(reloaded, "segment_global_threshold")


def test_cellpose_workflow_module_import_without_skfmm(monkeypatch):
    import scportrait.pipeline.segmentation.workflows._cellpose as cellpose_workflow

    monkeypatch.setitem(sys.modules, "skfmm", None)

    reloaded = importlib.reload(cellpose_workflow)

    assert hasattr(reloaded, "NuclearExpansionSegmentationCellpose")


def test_wga_workflow_module_import_without_skfmm(monkeypatch):
    import scportrait.pipeline.segmentation.workflows._wga_segmentation as wga_segmentation

    monkeypatch.setitem(sys.modules, "skfmm", None)

    reloaded = importlib.reload(wga_segmentation)

    assert hasattr(reloaded, "WGASegmentation")


def test_segment_global_threshold_missing_skfmm_raises_guided_error(monkeypatch):
    from scportrait.pipeline._utils.segmentation import segment_global_threshold

    monkeypatch.setitem(sys.modules, "skfmm", None)

    with pytest.raises(ImportError, match=r"scportrait\[segmentation\]"):
        segment_global_threshold(data.coins())
