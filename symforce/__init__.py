# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
The top-level symforce package

Importing this by itself performs minimal initialization configuration, and the functions here are
mostly for configuration purposes.

In particular, this primarily performs configuration that you might need before importing
:mod:`symforce.symbolic`.
"""

import os
import sys
import typing as T
from dataclasses import dataclass
from types import ModuleType

# -------------------------------------------------------------------------------------------------
# Version
# -------------------------------------------------------------------------------------------------

# isort: split
from ._version import version as __version__

# -------------------------------------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------------------------------------

# isort: split
import logging

# Create a logger with this print format
LOGGING_FORMAT = "%(module)s.%(funcName)s():%(lineno)s %(levelname)s -- %(message)s"
logging.basicConfig(format=LOGGING_FORMAT)
logger = logging.getLogger(__package__)


def set_log_level(log_level: str) -> None:
    """
    Set symforce logger level.

    The default is INFO, but can be set by one of:

    1) The SYMFORCE_LOGLEVEL environment variable
    2) Calling this function before any other symforce imports

    Args:
        log_level: {DEBUG, INFO, WARNING, ERROR, CRITICAL}
    """
    # Set default log level
    if not hasattr(logging, log_level.upper()):
        raise RuntimeError(f'Unknown log level: "{log_level}"')
    logger.setLevel(getattr(logging, log_level.upper()))

    # Only do this if already imported, in case users don't want to use any C++ binaries
    if "cc_sym" in sys.modules:
        import cc_sym

        cc_sym.set_log_level(log_level)


# Set default
set_log_level(os.environ.get("SYMFORCE_LOGLEVEL", "INFO"))

# -------------------------------------------------------------------------------------------------
# Symbolic API configuration
# -------------------------------------------------------------------------------------------------


class InvalidSymbolicApiError(Exception):
    def __init__(self, api: str):
        super().__init__(f'Symbolic API is "{api}", must be one of ("sympy", "symengine")')


def _find_symengine() -> ModuleType:
    """
    Attempts to import symengine from its location in the symforce build directory

    If symengine is already in sys.modules, will return that module.  If symengine cannot be
    imported, raises ImportError.

    Returns the imported symengine module
    """
    if "symengine" in sys.modules:
        return sys.modules["symengine"]

    try:
        # If symengine is available on python path, use it
        # TODO(will, aaron): this might not be the version of symengine that we want
        import symengine

        return symengine
    except ImportError as ex:
        import importlib
        import importlib.abc
        import importlib.util

        from . import path_util

        try:
            symengine_install_dir = path_util.symenginepy_install_dir()
        except path_util.MissingManifestException:
            raise ImportError(
                "Unable to import SymEngine, either installed or in the manifest.json"
            ) from ex

        symengine_path_candidates = list(
            symengine_install_dir.glob("lib/python3*/site-packages/symengine/__init__.py")
        ) + list(
            symengine_install_dir.glob("local/lib/python3*/dist-packages/symengine/__init__.py")
        )
        if len(symengine_path_candidates) != 1:
            raise ImportError(
                f"Should be exactly one symengine package, found candidates {symengine_path_candidates} in directory {path_util.symenginepy_install_dir()}"
            ) from ex
        symengine_path = symengine_path_candidates[0]

        # Import symengine from the directory where we installed it.  See
        # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
        spec = importlib.util.spec_from_file_location("symengine", symengine_path)
        assert spec is not None
        symengine = importlib.util.module_from_spec(spec)
        sys.modules["symengine"] = symengine

        # For mypy: https://github.com/python/typeshed/issues/2793
        assert isinstance(spec.loader, importlib.abc.Loader)

        try:
            spec.loader.exec_module(symengine)
        except:  # pylint: disable=bare-except
            # If executing the module fails for any reason, it shouldn't be in `sys.modules`
            del sys.modules["symengine"]
            raise

        return symengine


_symbolic_api: T.Optional[T.Literal["sympy", "symengine"]] = None
_have_imported_symbolic = False


def _set_symbolic_api(sympy_module: T.Literal["sympy", "symengine"]) -> None:
    # Set this as the default symbolic API
    global _symbolic_api  # pylint: disable=global-statement
    _symbolic_api = sympy_module


def _use_symengine() -> None:
    try:
        _find_symengine()

    except ImportError:
        logger.critical("Commanded to use symengine, but failed to import.")
        raise

    _set_symbolic_api("symengine")


def _use_sympy() -> None:
    # Import just to make sure it's importable and fail here if it's not (as opposed to failing
    # later)
    import sympy as sympy_py  # pylint: disable=unused-import

    _set_symbolic_api("sympy")


def set_symbolic_api(name: str) -> None:
    """
    Set the symbolic API for symforce

    See the SymPy tutorial for information on the symbolic APIs that can be used:
    https://symforce.org/tutorials/sympy_tutorial.html

    By default, SymForce will use the ``symengine`` API if it is available.  If the symbolic API is
    set to ``sympy`` it will use that.  If ``symengine`` is not available and the symbolic API was
    not set, it will emit a warning and use the ``sympy`` API.

    The symbolic API can be set by one of:

    1) The ``SYMFORCE_SYMBOLIC_API`` environment variable
    2) Calling this function before any other symforce imports

    Args:
        name: {sympy, symengine}
    """
    if _have_imported_symbolic and name != _symbolic_api:
        raise ValueError(
            "The symbolic API cannot be changed after `symforce.symbolic` has been imported.  "
            "Import the top-level `symforce` module and call `symforce.set_symbolic_api` before "
            "importing anything else!"
        )

    if _symbolic_api is not None and name == _symbolic_api:
        logger.debug(f'already on symbolic API "{name}"')
        return
    else:
        logger.debug(f'symbolic API: "{name}"')

    if name == "sympy":
        _use_sympy()
    elif name == "symengine":
        _use_symengine()
    else:
        raise NotImplementedError(f'Unknown symbolic API: "{name}"')


# Set default to symengine if available, else sympy
if "SYMFORCE_SYMBOLIC_API" in os.environ:
    set_symbolic_api(os.environ["SYMFORCE_SYMBOLIC_API"])
else:
    try:
        _find_symengine()

        logger.debug("No SYMFORCE_SYMBOLIC_API set, found and using symengine.")
        set_symbolic_api("symengine")
    except ImportError:
        logger.debug("No SYMFORCE_SYMBOLIC_API set, no symengine found.  Will use sympy.")
        pass


def get_symbolic_api() -> T.Literal["sympy", "symengine"]:
    """
    Return the current symbolic API as a string.
    """
    return _symbolic_api or "sympy"


# --------------------------------------------------------------------------------
# Default epsilon
# --------------------------------------------------------------------------------

# Should match C++ default epsilon in epsilon.h
numeric_epsilon = 10 * sys.float_info.epsilon


class AlreadyUsedEpsilon(Exception):
    """
    Exception thrown on attempting to modify the default epsilon after it has been used elsewhere
    """

    pass


_epsilon: T.Any = 0.0
_have_used_epsilon = False


def _set_epsilon(new_epsilon: T.Any) -> None:
    """
    Set the default epsilon for SymForce

    This must be called before :mod:`symforce.symbolic` or other symbolic libraries have been
    imported. Typically it should be set to some kind of Scalar, such as an int, float, or Symbol.
    See :func:`symforce.symbolic.epsilon` for more information.

    Args:
        new_epsilon: The new default epsilon to use
    """
    global _epsilon  # pylint: disable=global-statement

    if _have_used_epsilon and new_epsilon != _epsilon:
        raise AlreadyUsedEpsilon(
            f"Cannot set return value of epsilon to {new_epsilon} after it has already been "
            f"accessed with value {_epsilon}."
        )

    _epsilon = new_epsilon


@dataclass
class SymbolicEpsilon:
    """
    An indicator that SymForce should use a symbolic epsilon
    """

    name: str


def set_epsilon_to_symbol(name: str = "epsilon") -> None:
    """
    Set the default epsilon for Symforce to a Symbol.

    This must be called before :mod:`symforce.symbolic` or other symbolic libraries have been
    imported. See :func:`symforce.symbolic.epsilon` for more information.

    Args:
        name: The name of the symbol for the new default epsilon to use
    """
    _set_epsilon(SymbolicEpsilon(name))


def set_epsilon_to_number(value: T.Any = numeric_epsilon) -> None:
    """
    Set the default epsilon for Symforce to a number.

    This must be called before :mod:`symforce.symbolic` or other symbolic libraries have been
    imported. See :func:`symforce.symbolic.epsilon` for more information.

    Args:
        value: The new default epsilon to use
    """
    _set_epsilon(value)


def set_epsilon_to_zero() -> None:
    """
    Set the default epsilon for Symforce to zero.

    This must be called before :mod:`symforce.symbolic` or other symbolic libraries have been
    imported. See :func:`symforce.symbolic.epsilon` for more information.
    """
    _set_epsilon(0.0)


def set_epsilon_to_invalid() -> None:
    """
    Set the default epsilon for SymForce to ``None``.  Should not be used to actually create
    expressions or generate code.

    This is useful if you've forgotten to pass an epsilon somewhere, but are not sure where - using
    this epsilon in an expression should throw a ``TypeError`` near the location where you forgot to
    pass an epsilon.

    This must be called before :mod:`symforce.symbolic` or other symbolic libraries have been
    imported. See :func:`symforce.symbolic.epsilon` for more information.
    """
    _set_epsilon(None)
