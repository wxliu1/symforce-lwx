# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import json
import typing as T
from pathlib import Path


class MissingManifestException(RuntimeError):
    pass


class _Manifest:
    """
    Internal class to manage loading data from the build manifest and caching that data.  Not
    intended for use outside of path_util.py.
    """

    _manifest: T.Optional[T.Dict[str, str]] = None

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._manifest is None:
            manifest_path = Path(__file__).parent.parent / "build" / "manifest.json"
            try:
                with open(manifest_path) as f:
                    cls._manifest = json.load(f)
            except FileNotFoundError as ex:
                raise MissingManifestException(f"Manifest not found at {manifest_path}") from ex

    @classmethod
    def get_entry(cls, key: str) -> Path:
        cls._ensure_loaded()
        assert cls._manifest is not None
        return Path(cls._manifest[key]).resolve()

    @classmethod
    def get_entries(cls, key: str) -> T.List[Path]:
        cls._ensure_loaded()
        assert cls._manifest is not None
        return [Path(s).resolve() for s in cls._manifest[key]]


def symforce_dir() -> Path:
    return Path(__file__).parent.parent


def symenginepy_install_dir() -> Path:
    return _Manifest.get_entry("symenginepy_install_dir")


def cc_sym_install_dir() -> Path:
    return _Manifest.get_entry("cc_sym_install_dir")


def binary_output_dir() -> Path:
    return _Manifest.get_entry("binary_output_dir")


def symforce_root() -> Path:
    """
    The root directory of the symforce project
    """
    return Path(__file__).parent.parent


def symforce_data_root() -> Path:
    """
    The root directory of the symforce project, for use accessing data that might need to be updated
    (such as generated files).  Most of the time this is the same as :func:`symforce_root`, but when
    the ``--update`` flag is passed to a test, this is guaranteed to point to the *resolved*
    version, i.e. the actual symforce location on disk regardless of whether this path is a symlink.
    """
    from symforce.test_util.test_case_mixin import SymforceTestCaseMixin

    if SymforceTestCaseMixin.should_update():
        return Path(__file__).resolve().parent.parent
    else:
        return symforce_root()
