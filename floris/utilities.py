# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np


class Vec3():
    def __init__(self, x1, x2=None, x3=None, string_format=None):
        """
        Object containing vector information for coordinates.

        Args:
            x1: [float, float, float] or float -- The first argument
            can be a list of the three vector components or simply the
            first component of the vector.
            x2: float (optional) -- The second component of the vector.
            x3: float (optional) -- The third component of the vector.
            string_format: str (optional) -- The string format to use in the overloaded __str__ function.
        """
        if isinstance(x1, list):
            self.x1, self.x2, self.x3 = [float(x) for x in x1]
        else:
            self.x1 = float(x1)
            self.x2 = float(x2)
            self.x3 = float(x3)

        # TODO: checks:
        # - x1, x2, x3 are all of the same type

        if string_format is not None:
            self.string_format = string_format
        else:
            if type(self.x1) in [int]:
                self.string_format = "{:8d}"
            elif type(self.x1) in [float, np.float64]:
                self.string_format = "{:8.3f}"

    def rotate_on_x3(self, theta, center_of_rotation=None):
        """
        Rotate about the x3 coordinate axis by a given angle
        and center of rotation.
        The angle theta should be given in degrees.

        Sets the rotated components on this object and returns
        """
        if center_of_rotation is None:
            center_of_rotation = Vec3(0.0, 0.0, 0.0)
        x1offset = self.x1 - center_of_rotation.x1
        x2offset = self.x2 - center_of_rotation.x2
        self.x1prime = x1offset * cosd(theta) - x2offset * sind(
            theta) + center_of_rotation.x1
        self.x2prime = x2offset * cosd(theta) + x1offset * sind(
            theta) + center_of_rotation.x2
        self.x3prime = self.x3

    def __str__(self):
        template_string = "{} {} {}".format(self.string_format,
                                            self.string_format,
                                            self.string_format)
        return template_string.format(self.x1, self.x2, self.x3)

    def __add__(self, arg):

        if type(arg) is Vec3:
            return Vec3(self.x1 + arg.x1, self.x2 + arg.x2, self.x3 + arg.x3)
        else:
            return Vec3(self.x1 + arg, self.x2 + arg, self.x3 + arg)

    def __sub__(self, arg):
        if type(arg) is Vec3:
            return Vec3(self.x1 - arg.x1, self.x2 - arg.x2, self.x3 - arg.x3)
        else:
            return Vec3(self.x1 - arg, self.x2 - arg, self.x3 - arg)

    def __mul__(self, arg):

        if type(arg) is Vec3:
            return Vec3(self.x1 * arg.x1, self.x2 * arg.x2, self.x3 * arg.x3)
        else:
            return Vec3(self.x1 * arg, self.x2 * arg, self.x3 * arg)

    def __truediv__(self, arg):
        if type(arg) is Vec3:
            return Vec3(self.x1 / arg.x1, self.x2 / arg.x2, self.x3 / arg.x3)
        else:
            return Vec3(self.x1 / arg, self.x2 / arg, self.x3 / arg)

    def __eq__(self, arg):
        return self.x1 == arg.x1 \
            and self.x2 == arg.x2 \
            and self.x3 == arg.x3

    def __hash__(self):
        return hash((self.x1, self.x2, self.x3))


class Output():
    def __init__(self, filename):
        """
        Generate output file at specified location.

        Args:
            filename ([str]): write to location
        """
        self.filename = filename
        self.file = open(self.filename, "w")
        self.ln = "\n"

    def write_empty_line(self):
        """
        Add blank line to file.
        """
        self.write_line("")

    def write_line(self, line):
        """
        Add line to file with specified content.

        Args:
            line ([str]): content to add to file
        """
        self.file.write(line + self.ln)

    def end(self):
        """
        Close file.
        """
        self.file.close()


def cosd(angle):
    """
    cosine of an angle with the angle given in degrees

    Returns:
        [float]: cosine of angle in degrees
    """
    return np.cos(np.radians(angle))


def sind(angle):
    """
    sine of an angle with the angle given in degrees

    Returns:
        [float]: sine of angle in degrees
    """
    return np.sin(np.radians(angle))


def tand(angle):
    """
    tangent of an angle with the angle given in degrees

    Returns:
        [float]: tangent of angle in degrees
    """
    return np.tan(np.radians(angle))


def wrap_180(x):
    """
    Wrap an angle to between -180 and 180

    Returns:
        [array]: angles in specified interval
    """
    x = np.where(x <= -180., x + 360., x)
    x = np.where(x > 180., x - 360., x)
    return (x)


def wrap_360(x):
    """
    Wrap an angle to between 0 and 360

    Returns:
        [array]: angles in specified interval
    """
    x = np.where(x < 0., x + 360., x)
    x = np.where(x >= 360., x - 360., x)
    return (x)
