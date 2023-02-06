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

import os
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


def wind_delta(wind_directions):
    """
    This is included as a function in order to facilitate testing.
    """
    return ((wind_directions - 270) % 360 + 360) % 360


def rotate_coordinates_rel_west(wind_directions, coordinates):
    # Calculate the difference in given wind direction from 270 / West
    wind_deviation_from_west = wind_delta(wind_directions)
    wind_deviation_from_west = np.reshape(wind_deviation_from_west, (len(wind_directions), 1, 1))

    # Construct the arrays storing the turbine locations
    x_coordinates, y_coordinates, z_coordinates = coordinates.T

    # Find center of rotation - this is the center of box bounding all of the turbines
    x_center_of_rotation = (np.min(x_coordinates) + np.max(x_coordinates)) / 2
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
    return x_coord_rotated, y_coord_rotated, z_coord_rotated


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
    if isinstance(filename, dict):
        return filename  # filename already yaml dict
    with open(filename) as fid:
        return yaml.load(fid, loader)
