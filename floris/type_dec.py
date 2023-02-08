# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Tuple,
    Union,
)

import attrs
import numpy as np
import numpy.typing as npt
from attrs import Attribute, define


### Define general data types used throughout

floris_float_type = np.float64

NDArrayFloat = npt.NDArray[floris_float_type]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFilter = Union[npt.NDArray[np.int_], npt.NDArray[np.bool_]]
NDArrayObject = npt.NDArray[np.object_]


### Custom callables for attrs objects and functions

def floris_array_converter(data: Iterable) -> np.ndarray:
    try:
        a = np.array(data, dtype=floris_float_type)
    except TypeError as e:
        raise TypeError(e.args[0] + f". Data given: {data}")
    return a

def attr_serializer(inst: type, field: Attribute, value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value

def attr_floris_filter(inst: Attribute, value: Any) -> bool:
    if inst.init is False:
        return False
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
    return True

def iter_validator(iter_type, item_types: Union[Any, Tuple[Any]]) -> Callable:
    """Helper function to generate iterable validators that will reduce the amount of
    boilerplate code.

    Parameters
    ----------
    iter_type : any iterable
        The type of iterable object that should be validated.
    item_types : Union[Any, Tuple[Any]]
        The type or types of acceptable item types.

    Returns
    -------
    Callable
        The attr.validators.deep_iterable iterable and instance validator.
    """
    validator = attrs.validators.deep_iterable(
        member_validator=attrs.validators.instance_of(item_types),
        iterable_validator=attrs.validators.instance_of(iter_type),
    )
    return validator


def convert_to_path(fn: str | Path) -> Path:
    """Converts an input string or pathlib.Path object to a fully resolved ``pathlib.Path``
    object.

    Args:
        fn (str | Path): The user input file path or file name.

    Raises:
        TypeError: Raised if :py:attr:`fn` is neither a :py:obj:`str`, nor a :py:obj:`pathlib.Path`.

    Returns:
        Path: A resolved pathlib.Path object.
    """
    if isinstance(fn, str):
        fn = Path(fn)

    if isinstance(fn, Path):
        fn.resolve()
    else:
        raise TypeError(f"The passed input: {fn} could not be converted to a pathlib.Path object")
    return fn


@define
class FromDictMixin:
    """
    A Mixin class to allow for kwargs overloading when a data class doesn't
    have a specific parameter definied. This allows passing of larger dictionaries
    to a data class without throwing an error.
    """

    @classmethod
    def from_dict(cls, data: dict):
        """Maps a data dictionary to an `attr`-defined class.

        TODO: Add an error to ensure that either none or all the parameters are passed in

        Args:
            data : dict
                The data dictionary to be mapped.
        Returns:
            cls
                The `attr`-defined class.
        """
        # Check for any inputs that aren't part of the class definition
        class_attr_names = [a.name for a in cls.__attrs_attrs__]
        extra_args = [d for d in data if d not in class_attr_names]
        if len(extra_args):
            raise AttributeError(
                f"The initialization for {cls.__name__} was given extraneous inputs: {extra_args}"
            )

        kwargs = {a.name: data[a.name] for a in cls.__attrs_attrs__ if a.name in data and a.init}

        # Map the inputs must be provided: 1) must be initialized, 2) no default value defined
        required_inputs = [
            a.name
            for a in cls.__attrs_attrs__
            if a.init and a.default is attrs.NOTHING
        ]
        undefined = sorted(set(required_inputs) - set(kwargs))

        if undefined:
            raise AttributeError(
                f"The class defintion for {cls.__name__} "
                "is missing the following inputs: {undefined}"
            )
        return cls(**kwargs)

    def as_dict(self) -> dict:
        """Creates a JSON and YAML friendly dictionary that can be save for future reloading.
        This dictionary will contain only `Python` types that can later be converted to their
        proper `Turbine` formats.

        Returns:
            dict: All key, vaue pais required for class recreation.
        """
        return attrs.asdict(self, filter=attr_floris_filter, value_serializer=attr_serializer)


# Avoids constant redefinition of the same attr.ib properties for model attributes

# from functools import partial, update_wrapper

# def is_default(instance, attribute, value):
#     if attribute.default != value:
#         raise ValueError(f"{attribute.name} should never be set manually.")

# model_attrib = partial(field, on_setattr=attrs.setters.frozen, validator=is_default)
# update_wrapper(model_attrib, field)

# float_attrib = partial(
#     attr.ib,
#     converter=float,
#     on_setattr=(attr.setters.convert, attr.setters.validate),  # type: ignore
#     kw_only=True,
# )
# update_wrapper(float_attrib, attr.ib)

# bool_attrib = partial(
#     attr.ib,
#     converter=bool,
#     on_setattr=(attr.setters.convert, attr.setters.validate),  # type: ignore
#     kw_only=True,
# )
# update_wrapper(bool_attrib, attr.ib)

# int_attrib = partial(
#     attr.ib,
#     converter=int,
#     on_setattr=(attr.setters.convert, attr.setters.validate),  # type: ignore
#     kw_only=True,
# )
# update_wrapper(int_attrib, attr.ib)
