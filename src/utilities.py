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

from typing import Any, Dict, List, Tuple, Union

import attr
import numpy as np


class Vec3:
    def __init__(self, components: List[float]):
        """
        Contains 3-component vector information. All arithmetic operators are
        set so that Vec3 objects can operate on and with each other directly.

        Args:
            x1 (list(numeric, numeric, numeric), numeric): All three vector
                components or simply the first component of the vector.
            x2 (numeric, optional): Second component of the vector if the
                first argument is not a list. Defaults to None.
            x3 (numeric, optional): Third component of the vector if the
                first argument is not a list. Defaults to None.
            string_format (str, optional): Format to use in the
                overloaded __str__ function. Defaults to None.
        """

        # TODO: possibility to store components as np.array or Python list
        # and use x1, x2, x3 virtual properties. This would simplify the
        # arithmetic overloading by allowing the use of numpy array
        # operations. It may add some performance gain, too, but that is
        # likely negligible.

        if len(components) != 3:
            raise TypeError(
                "Vec3 requires 3 components, {} given.".format(len(components))
            )

        self.components = [float(c) for c in components]

    def rotate_on_x3(self, theta, center_of_rotation=None):
        """
        Rotates about the `x3` coordinate axis by a given angle
        and center of rotation. This function sets additional attributes on
        the rotated Vec3:

            - x1prime
            - x2prime
            - x3prime

        Args:
            theta (float): Angle of rotation in degrees.
            center_of_rotation (Vec3, optional): Center of rotation.
                Defaults to Vec3(0.0, 0.0, 0.0).
        """
        if center_of_rotation is None:
            center_of_rotation = Vec3([0.0, 0.0, 0.0])
        x1offset = self.x1 - center_of_rotation.x1
        x2offset = self.x2 - center_of_rotation.x2
        self.x1prime = (
            x1offset * cosd(theta) - x2offset * sind(theta) + center_of_rotation.x1
        )
        self.x2prime = (
            x2offset * cosd(theta) + x1offset * sind(theta) + center_of_rotation.x2
        )
        self.x3prime = self.x3

    def __str__(self):
        return f"{self.x1:8.3f} {self.xs:8.3f} {self.x3:8.3f}"

    def __add__(self, arg):
        if type(arg) is Vec3:
            return Vec3([self.x1 + arg.x1, self.x2 + arg.x2, self.x3 + arg.x3])
        else:
            return Vec3([self.x1 + arg, self.x2 + arg, self.x3 + arg])

    def __sub__(self, arg):
        if type(arg) is Vec3:
            return Vec3([self.x1 - arg.x1, self.x2 - arg.x2, self.x3 - arg.x3])
        else:
            return Vec3([self.x1 - arg, self.x2 - arg, self.x3 - arg])

    def __mul__(self, arg):
        if type(arg) is Vec3:
            return Vec3([self.x1 * arg.x1, self.x2 * arg.x2, self.x3 * arg.x3])
        else:
            return Vec3([self.x1 * arg, self.x2 * arg, self.x3 * arg])

    def __truediv__(self, arg):
        if type(arg) is Vec3:
            return Vec3([self.x1 / arg.x1, self.x2 / arg.x2, self.x3 / arg.x3])
        else:
            return Vec3([self.x1 / arg, self.x2 / arg, self.x3 / arg])

    def __eq__(self, arg):
        return False not in np.isclose(
            [self.x1, self.x2, self.x3], [arg.x1, arg.x2, arg.x3]
        )

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

    @x2.setter
    def x2(self, value):
        self.components[2] = float(value)

    @property
    def elements(self) -> Tuple[float, float, float]:
        return self.x1, self.x2, self.x3

    @property
    def prime_elements(self) -> Tuple[float, float, float]:
        return self.x1prime, self.x2prime, self.x3prime


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
        np.array: Shifted values.
    """
    x = np.where(x <= -180.0, x + 360.0, x)
    x = np.where(x > 180.0, x - 360.0, x)
    return x


def wrap_360(x):
    """
    Shift the given values to within the range (0, 360].

    Args:
        x (numeric or np.array): Scalar value or np.array of values to shift.

    Returns:
        np.array: Shifted values.
    """
    x = np.where(x < 0.0, x + 360.0, x)
    x = np.where(x >= 360.0, x - 360.0, x)
    return x


def convert_to_Vec3(x: Union[List[float], Vec3]) -> Vec3:
    if isinstance(x, Vec3):
        return x
    return Vec3(x)


def is_default(instance, attribute, value):
    if attribute.default != value:
        raise ValueError(f"{attribute.name} should never be set manually!")


@attr.s
class FromDictMixin:
    """A Mixin class to allow for kwargs overloading when a data class doesn't
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
        return cls(
            **{a.name: data[a.name] for a in cls.__attrs_attrs__ if a.name in data}
        )

    @classmethod
    def get_model_defaults(cls) -> Dict[str, Any]:
        """Produces a dictionary of the keyword arguments and their defaults.

        Returns
        -------
        Dict[str, Any]
            Dictionary of keyword argument: default.
        """
        return {el.name: el.default for el in attr.fields(cls)}
