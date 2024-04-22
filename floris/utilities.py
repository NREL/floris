
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

from floris.type_dec import floris_array_converter, NDArrayFloat


def pshape(array: np.ndarray, label: str = ""):
    print(label, np.shape(array))


def cosd(angle):
    """
    Cosine of an angle with the angle given in degrees.

    Args:
        angle (float): Angle in degrees.

    Returns:
        float
    """
    return np.cos(np.radians(angle))


def sind(angle):
    """
    Sine of an angle with the angle given in degrees.

    Args:
        angle (float): Angle in degrees.

    Returns:
        float
    """
    return np.sin(np.radians(angle))


def tand(angle):
    """
    Tangent of an angle with the angle given in degrees.

    Args:
        angle (float): Angle in degrees.

    Returns:
        float
    """
    return np.tan(np.radians(angle))


def wrap_180(x):
    """
    Shift the given values to within the range (-180, 180].

    Args:
        x (numeric or np.array): Scalar value or np.array of values to shift.

    Returns:
        np.ndarray | float | int: Shifted values.
    """

    return ((x + 180.0) % 360.0) - 180.0


def wrap_360(x):
    """
    Shift the given values to within the range (0, 360].

    Args:
        x (numeric or np.array): Scalar value or np.array of values to shift.

    Returns:
        np.ndarray | float | int: Shifted values.
    """

    return x % 360.0


def wind_delta(wind_directions: NDArrayFloat | float):
    """
    This function calculates the deviation from West (270) for a given wind direction or series
    of wind directions. First, 270 is subtracted from the input wind direction, and then the
    remainder after dividing by 360 is retained (modulo). The table below lists examples of
    results.

    | Input | Result |
    | ----- | ------ |
    | 270.0 | 0.0    |
    | 280.0 | 10.0   |
    | 360.0 | 90.0   |
    | 180.0 | 270.0  |
    | -10.0 | 80.0   |
    |-100.0 | 350.0  |

    Args:
        wind_directions (NDArrayFloat | float): A single or series of wind directions. They can be
        any number, negative or positive, but it is typically between 0 and 360.

    Returns:
        NDArrayFloat | float: The delta between the given wind direction and 270 in positive
        quantities between 0 and 360. The returned type is the same as wind_directions.
    """

    return (wind_directions - 270) % 360


def rotate_coordinates_rel_west(
    wind_directions,
    coordinates,
    x_center_of_rotation=None,
    y_center_of_rotation=None
):
    """
    This function rotates the given coordinates so that they are aligned with West (270) rather
    than North (0). The rotation happens about the centroid of the coordinates.

    Args:
        wind_directions (NDArrayFloat): Series of wind directions to base the rotation.
        coordinates (NDArrayFloat): Series of coordinates to rotate with shape (N coordinates, 3)
            so that each element of the array coordinates[i] yields a three-component coordinate.
        x_center_of_rotation (float, optional): The x-coordinate for the rotation center of the
            input coordinates. Defaults to None.
        y_center_of_rotation (float, optional): The y-coordinate for the rotational center of the
            input coordinates. Defaults to None.
    """

    # Calculate the difference in given wind direction from 270 / West
    wind_deviation_from_west = wind_delta(wind_directions)
    wind_deviation_from_west = np.reshape(wind_deviation_from_west, (len(wind_directions), 1))

    # Construct the arrays storing the turbine locations
    x_coordinates, y_coordinates, z_coordinates = coordinates.T

    # Find center of rotation - this is the center of box bounding all of the turbines
    if x_center_of_rotation is None:
        x_center_of_rotation = (np.min(x_coordinates) + np.max(x_coordinates)) / 2
    if y_center_of_rotation is None:
        y_center_of_rotation = (np.min(y_coordinates) + np.max(y_coordinates)) / 2

    # Rotate turbine coordinates about the center
    x_coord_offset = x_coordinates - x_center_of_rotation
    y_coord_offset = y_coordinates - y_center_of_rotation
    x_coord_rotated = (
        x_coord_offset * cosd(wind_deviation_from_west)
        - y_coord_offset * sind(wind_deviation_from_west)
        + x_center_of_rotation
    )
    y_coord_rotated = (
        x_coord_offset * sind(wind_deviation_from_west)
        + y_coord_offset * cosd(wind_deviation_from_west)
        + y_center_of_rotation
    )
    z_coord_rotated = np.ones_like(wind_deviation_from_west) * z_coordinates
    return x_coord_rotated, y_coord_rotated, z_coord_rotated, x_center_of_rotation, \
        y_center_of_rotation


def reverse_rotate_coordinates_rel_west(
    wind_directions: NDArrayFloat,
    grid_x: NDArrayFloat,
    grid_y: NDArrayFloat,
    grid_z: NDArrayFloat,
    x_center_of_rotation: float,
    y_center_of_rotation: float
):
    """
    This function reverses the rotation of the given grid so that the coordinates are aligned with
    the original wind direction. The rotation happens about the centroid of the coordinates.

    Args:
        wind_directions (NDArrayFloat): Series of wind directions to base the rotation.
        grid_x (NDArrayFloat): X-coordinates to be rotated.
        grid_y (NDArrayFloat): Y-coordinates to be rotated.
        grid_z (NDArrayFloat): Z-coordinates to be rotated.
        x_center_of_rotation (float): The x-coordinate for the rotation center of the
            input coordinates.
        y_center_of_rotation (float): The y-coordinate for the rotational center of the
            input coordinates.
    """
    # Calculate the difference in given wind direction from 270 / West
    # We are rotating in the other direction
    wind_deviation_from_west = -1.0 * wind_delta(wind_directions)

    # Construct the arrays storing the turbine locations
    grid_x_reversed = np.zeros_like(grid_x)
    grid_y_reversed = np.zeros_like(grid_x)
    grid_z_reversed = np.zeros_like(grid_x)
    for wii, angle_rotation in enumerate(wind_deviation_from_west):
        x_rot = grid_x[wii]
        y_rot = grid_y[wii]
        z_rot = grid_z[wii]

        # Rotate turbine coordinates about the center
        x_rot_offset = x_rot - x_center_of_rotation
        y_rot_offset = y_rot - y_center_of_rotation
        x = (
            x_rot_offset * cosd(angle_rotation)
            - y_rot_offset * sind(angle_rotation)
            + x_center_of_rotation
        )
        y = (
            x_rot_offset * sind(angle_rotation)
            + y_rot_offset * cosd(angle_rotation)
            + y_center_of_rotation
        )
        z = z_rot  # Nothing changed in this rotation

        grid_x_reversed[wii] = x
        grid_y_reversed[wii] = y
        grid_z_reversed[wii] = z

    return grid_x_reversed, grid_y_reversed, grid_z_reversed


class Loader(yaml.SafeLoader):

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super().__init__(stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
            return yaml.load(f, self.__class__)


Loader.add_constructor('!include', Loader.include)

def load_yaml(filename, loader=Loader):
    with open(filename) as fid:
        return yaml.load(fid, loader)


def round_nearest_2_or_5(x: int | float) -> int:
    """Rounds a number (with a 0.5 buffer) up to the nearest integer divisible by 2 or 5.

    Args:
        x (int | float): The number to be rounded.

    Returns:
        int: The rounded number.
    """
    base_2 = 2
    base_5 = 5
    return min(base_2 * ceil((x + 0.5) / base_2), base_5 * ceil((x + 0.5) / base_5))


def round_nearest(x: int | float, base: int = 5) -> int:
    """Rounds a number (with a 0.5 buffer) up to the nearest integer divisible by 5.

    Args:
        x (int | float): The number to be rounded.

    Returns:
        int: The rounded number.
    """
    return base * ceil((x + 0.5) / base)


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
