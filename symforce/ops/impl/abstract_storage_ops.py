# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import abc

from symforce import typing as T

ElementT = T.TypeVar("ElementT")
ElementOrTypeT = T.Union[ElementT, T.Type[ElementT]]


class AbstractStorageOps(abc.ABC, T.Generic[ElementT]):
    """
    An abstract base class for StorageOps implementations.

    Useful for when multiple classes can implement their GroupOps and LieGroupOps
    implementations in terms of their storage ops in the same manner, despite having
    different StorageOps impelmentations.

    For example, classes whose storage spaces are abstract vector spaces and whose
    group operations are their vector operations. See :mod:`.abstract_vector_group_ops`.
    """

    # NOTE(aaron): These should be @staticmethods, this is fixed in mypy 1.9
    # https://github.com/python/mypy/pull/16670

    @classmethod
    @abc.abstractmethod
    def storage_dim(cls, a: ElementOrTypeT) -> int:
        pass

    @classmethod
    @abc.abstractmethod
    def to_storage(cls, a: ElementT) -> T.List[T.Scalar]:
        pass

    @classmethod
    @abc.abstractmethod
    def from_storage(cls, a: ElementOrTypeT, elements: T.Sequence[T.Scalar]) -> ElementT:
        pass

    @classmethod
    @abc.abstractmethod
    def symbolic(cls, a: ElementOrTypeT, name: str, **kwargs: T.Dict) -> ElementT:
        pass
