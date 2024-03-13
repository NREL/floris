
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


def nested_get(
    d: Dict[str, Any],
    keys: List[str]
) -> Any:
    """Get a value from a nested dictionary using a list of keys.
    Based on:
    https://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys

    Args:
        d (Dict[str, Any]): The dictionary to get the value from.
        keys (List[str]): A list of keys to traverse the dictionary.

    Returns:
        Any: The value at the end of the key traversal.
    """
    for key in keys:
        d = d[key]
    return d

def nested_set(
    d: Dict[str, Any],
    keys: List[str],
    value: Any,
    idx: Optional[int] = None
) -> None:
    """Set a value in a nested dictionary using a list of keys.
    Based on:
    https://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys

    Args:
        dic (Dict[str, Any]): The dictionary to set the value in.
        keys (List[str]): A list of keys to traverse the dictionary.
        value (Any): The value to set.
        idx (Optional[int], optional): If the value is an list, the index to change.
            Defaults to None.
    """
    d_in = d.copy()

    for key in keys[:-1]:
        d = d.setdefault(key, {})
    if idx is None:
        # Parameter is a scalar, set directly
        d[keys[-1]] = value
    else:
        # Parameter is a list, need to first get the list, change the values at idx

        # # Get the underlying list
        par_list = nested_get(d_in, keys)
        par_list[idx] = value
        d[keys[-1]] = par_list

def print_nested_dict(dictionary: Dict[str, Any], indent: int = 0) -> None:
    """Print a nested dictionary with indentation.

    Args:
        dictionary (Dict[str, Any]): The dictionary to print.
        indent (int, optional): The number of spaces to indent. Defaults to 0.
    """
    for key, value in dictionary.items():
        print(" " * indent + str(key))
        if isinstance(value, dict):
            print_nested_dict(value, indent + 4)
        else:
            print(" " * (indent + 4) + str(value))

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
