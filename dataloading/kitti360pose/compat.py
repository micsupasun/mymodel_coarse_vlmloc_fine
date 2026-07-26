"""Narrow compatibility helpers for legacy KITTI360Pose pickle module names."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, BinaryIO


_LEGACY_MODULE_PREFIX = "datapreparation.kitti360"
_CURRENT_MODULE_PREFIX = "datapreparation.kitti360pose"


class Kitti360PoseUnpickler(pickle.Unpickler):
    """Remap only the historical package name used by released pickle files.

    The classes themselves are unchanged; this does not alter serialized values
    or install a process-wide ``sys.modules`` alias.
    """

    def find_class(self, module: str, name: str) -> Any:
        if module == _LEGACY_MODULE_PREFIX or module.startswith(
            f"{_LEGACY_MODULE_PREFIX}."
        ):
            module = f"{_CURRENT_MODULE_PREFIX}{module[len(_LEGACY_MODULE_PREFIX):]}"
        return super().find_class(module, name)


def load_kitti360pose_pickle(path_or_file: str | Path | BinaryIO) -> Any:
    """Load one dataset pickle with the localized legacy-module remapping."""

    if hasattr(path_or_file, "read"):
        return Kitti360PoseUnpickler(path_or_file).load()

    with Path(path_or_file).open("rb") as handle:
        return Kitti360PoseUnpickler(handle).load()
