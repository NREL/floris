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

import os
from math import ceil
from typing import Tuple

import numpy as np
import yaml
from attrs import define, field

from floris.type_dec import floris_array_converter, NDArrayFloat


def pshape(array: np.ndarray, label: str = ""):
    print(label, np.shape(array))


@define
class Vec3:
    """
    Contains 3-component vector information. All arithmetic operators are
    set so that Vec3 objects can operate on and with each other directly.

    Args:
        components (list(numeric, numeric, numeric), numeric): All three vector
            components.
        string_format (str, optional): Format to use in the
            overloaded __str__ function. Defaults to None.
    """
    components: NDArrayFloat = field(converter=floris_array_converter)
    # NOTE: this does not convert elements to float if they are given as int. Is this ok?

    @components.validator
    def _check_components(self, attribute, value) -> None:
        if np.ndim(value) > 1:
            raise ValueError(
                f"Vec3 must contain exactly 1 dimension, {np.ndim(value)} were given."
            )
        if np.size(value) != 3:
            raise ValueError(
                f"Vec3 must contain exactly 3 components, {np.size(value)} were given."
            )

    def __add__(self, arg):
        if type(arg) is Vec3:
            return Vec3(self.components + arg.components)
        elif type(arg) is int or type(arg) is float:
            return Vec3(self.components + arg)
        else:
            raise ValueError

    def __sub__(self, arg):
        if type(arg) is Vec3:
            return Vec3(self.components - arg.components)
        elif type(arg) is int or type(arg) is float:
            return Vec3(self.components - arg)
        else:
            raise ValueError

    def __mul__(self, arg):
        if type(arg) is Vec3:
            return Vec3(self.components * arg.components)
        elif type(arg) is int or type(arg) is float:
            return Vec3(self.components * arg)
        else:
            raise ValueError

    def __truediv__(self, arg):
        if type(arg) is Vec3:
            return Vec3(self.components / arg.components)
        elif type(arg) is int or type(arg) is float:
            return Vec3(self.components / arg)
        else:
            raise ValueError

    def __eq__(self, arg):
        return False not in np.isclose([self.x1, self.x2, self.x3], [arg.x1, arg.x2, arg.x3])

    def __hash__(self):
        return hash((self.x1, self.x2, self.x3))

    @property
    def x1(self):
        return self.components[0]

    @x1.setter
    def x1(self, value):
        self.components[0] = float(value)

    @property
    def x2(self):
        return self.components[1]

    @x2.setter
    def x2(self, value):
        self.components[1] = float(value)

    @property
    def x3(self):
        return self.components[2]

    @x3.setter
    def x3(self, value):
        self.components[2] = float(value)

    @property
    def elements(self) -> Tuple[float, float, float]:
        # TODO: replace references to elements with components
        # and remove this @property
        return self.components


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
    wind_deviation_from_west = np.reshape(wind_deviation_from_west, (len(wind_directions), 1, 1))

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
        coordinates (NDArrayFloat): Series of coordinates to rotate with shape (N coordinates, 3)
            so that each element of the array coordinates[i] yields a three-component coordinate.
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
        x_rot = grid_x[wii, :, :, :, :]
        y_rot = grid_y[wii, :, :, :, :]
        z_rot = grid_z[wii, :, :, :, :]

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

        grid_x_reversed[wii, :, :, :, :] = x
        grid_y_reversed[wii, :, :, :, :] = y
        grid_z_reversed[wii, :, :, :, :] = z

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
