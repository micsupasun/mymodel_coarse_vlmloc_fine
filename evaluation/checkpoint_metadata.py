"""Read tensor keys/shapes from PyTorch ZIP checkpoints without importing torch.

This is a static inspection helper only. Runtime preflight still constructs the
real architecture and uses :mod:`evaluation.checkpoint_loading`.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import zipfile
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_STORAGE_DTYPES = {
    "BFloat16Storage": "torch.bfloat16",
    "BoolStorage": "torch.bool",
    "ByteStorage": "torch.uint8",
    "CharStorage": "torch.int8",
    "ComplexDoubleStorage": "torch.complex128",
    "ComplexFloatStorage": "torch.complex64",
    "DoubleStorage": "torch.float64",
    "FloatStorage": "torch.float32",
    "HalfStorage": "torch.float16",
    "IntStorage": "torch.int32",
    "LongStorage": "torch.int64",
    "ShortStorage": "torch.int16",
}


@dataclass
class _StorageType:
    name: str


@dataclass
class _Storage:
    dtype: str
    size: int


@dataclass
class _Tensor:
    shape: tuple[int, ...]
    dtype: str


def _rebuild_tensor(storage: _Storage, _offset: int, size: Any, *_args: Any) -> _Tensor:
    return _Tensor(tuple(int(value) for value in size), storage.dtype)


def _rebuild_parameter(tensor: _Tensor, *_args: Any) -> _Tensor:
    return tensor


class _MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name in {
            "_rebuild_tensor",
            "_rebuild_tensor_v2",
            "_rebuild_tensor_v3",
        }:
            return _rebuild_tensor
        if module == "torch._utils" and name in {
            "_rebuild_parameter",
            "_rebuild_parameter_with_state",
        }:
            return _rebuild_parameter
        if module == "torch" and name.endswith("Storage"):
            return _StorageType(name)
        raise pickle.UnpicklingError(
            f"unsupported global in static checkpoint metadata: {module}.{name}"
        )

    def persistent_load(self, persistent_id: Any) -> _Storage:
        if (
            not isinstance(persistent_id, tuple)
            or len(persistent_id) < 5
            or persistent_id[0] != "storage"
        ):
            raise pickle.UnpicklingError(
                f"unsupported persistent id: {persistent_id!r}"
            )
        storage_type = persistent_id[1]
        storage_size = int(persistent_id[4])
        if not isinstance(storage_type, _StorageType):
            raise pickle.UnpicklingError(
                f"unsupported storage type: {storage_type!r}"
            )
        dtype = _STORAGE_DTYPES.get(storage_type.name)
        if dtype is None:
            raise pickle.UnpicklingError(
                f"unknown storage dtype: {storage_type.name}"
            )
        return _Storage(dtype, storage_size)


def inspect_pytorch_zip_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with zipfile.ZipFile(path) as archive:
        pickle_names = [
            name for name in archive.namelist() if name.endswith("/data.pkl")
        ]
        if len(pickle_names) != 1:
            raise ValueError(
                f"expected one */data.pkl in {path}, found {pickle_names}"
            )
        payload = archive.read(pickle_names[0])
    state = _MetadataUnpickler(io.BytesIO(payload)).load()
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint root is {type(state).__name__}, not a state dict")
    tensors = {
        key: {"shape": list(value.shape), "dtype": value.dtype}
        for key, value in state.items()
        if isinstance(key, str) and isinstance(value, _Tensor)
    }
    if len(tensors) != len(state):
        raise TypeError(
            f"state dict has {len(state) - len(tensors)} non-tensor entries"
        )
    return {
        "path": str(path),
        "key_count": len(tensors),
        "prefix_counts": dict(
            sorted(Counter(key.split(".", 1)[0] for key in tensors).items())
        ),
        "tensors": tensors,
    }


def compare_checkpoint_metadata(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    left_tensors = left["tensors"]
    right_tensors = right["tensors"]
    left_keys = set(left_tensors)
    right_keys = set(right_tensors)
    shape_mismatches = []
    for key in sorted(left_keys & right_keys):
        if left_tensors[key]["shape"] != right_tensors[key]["shape"]:
            shape_mismatches.append(
                {
                    "key": key,
                    "left_shape": left_tensors[key]["shape"],
                    "right_shape": right_tensors[key]["shape"],
                }
            )
    return {
        "left_path": left["path"],
        "right_path": right["path"],
        "left_key_count": left["key_count"],
        "right_key_count": right["key_count"],
        "only_left": sorted(left_keys - right_keys),
        "only_right": sorted(right_keys - left_keys),
        "shape_mismatches": shape_mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    reports = [
        inspect_pytorch_zip_checkpoint(path) for path in args.checkpoints
    ]
    payload: dict[str, Any] = {"checkpoints": reports}
    if args.compare:
        if len(reports) != 2:
            parser.error("--compare requires exactly two checkpoints")
        payload["comparison"] = compare_checkpoint_metadata(*reports)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
