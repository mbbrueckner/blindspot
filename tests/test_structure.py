"""The old module paths (``blindspot.evaluation``, ``blindspot.sanpo``, ...) re-export the new ones."""

import importlib

import pytest

from blindspot import calibration, evaluation, sanpo, splits
from blindspot.data import annotations
from blindspot.data import calibration as new_calibration
from blindspot.data import sanpo as new_sanpo
from blindspot.data import splits as new_splits
from blindspot.eval import metrics, runner


@pytest.mark.parametrize(
    "old, new",
    [
        (evaluation, [metrics, runner, annotations]),
        (sanpo, [new_sanpo]),
        (calibration, [new_calibration]),
        (splits, [new_splits]),
    ],
)
def test_compat_modules_reexport_the_same_objects(old, new):
    assert old.__all__
    for name in old.__all__:
        source = next(m for m in new if hasattr(m, name))
        assert getattr(old, name) is getattr(source, name), name


@pytest.mark.parametrize("name", ["data", "eval", "models", "alerts", "viz"])
def test_subpackages_import(name):
    assert importlib.import_module(f"blindspot.{name}").__doc__
