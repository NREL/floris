
from __future__ import annotations

import os
from math import ceil
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
import yaml
from attrs import define, field

from floris import FlorisModel
from floris.type_dec import floris_array_converter, NDArrayFloat
from floris.utilities import (
    nested_get,
    nested_set,
    print_nested_dict,
)


def print_fmodel_dict(fmodel: FlorisModel) -> None:
    """Print the FlorisModel dictionary.

    Args:
        fmodel (FlorisModel): The FlorisModel to print.
    """
    print_nested_dict(fmodel.core.as_dict())

def set_fmodel_param(
    fmodel_in: FlorisModel,
    param: List[str],
    value: Any,
    param_idx: Optional[int] = None
):
    """Set a parameter in a FlorisModel object.

    Args:
        fi_in (FlorisModel): The FlorisModel object to modify.
        param (List[str]): A list of keys to traverse the FlorisModel dictionary.
        value (Any): The value to set.
        idx (Optional[int], optional): The index to set the value at. Defaults to None.

    Returns:
        FlorisModel: The modified FlorisModel object.
    """
    fm_dict_mod = fmodel_in.core.as_dict()
    nested_set(fm_dict_mod, param, value, param_idx)
    return FlorisModel(fm_dict_mod)

def get_fmodel_param(
    fmodel_in: FlorisModel,
    param: List[str],
    param_idx: Optional[int] = None
) -> Any:
    """Get a parameter from a FlorisModel object.

    Args:
        fmodel_in (FlorisModel): The FlorisModel object to get the parameter from.
        param (List[str]): A list of keys to traverse the FlorisModel dictionary.
        param_idx (Optional[int], optional): The index to get the value at. Defaults to None.
            If None, the entire parameter is returned.

    Returns:
        Any: The value of the parameter.
    """
    fm_dict = fmodel_in.core.as_dict()

    if param_idx is None:
        return nested_get(fm_dict, param)
    else:
        return nested_get(fm_dict, param)[param_idx]
