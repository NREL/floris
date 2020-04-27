# Copyright 2020 NREL
 
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0
 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
 
# See https://floris.readthedocs.io for documentation
 

import numpy as np
import coloredlogs
import logging
from datetime import datetime


class Vec3():
    def __init__(self, x1, x2=None, x3=None, string_format=None):
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

        if isinstance(x1, list):
            self.x1, self.x2, self.x3 = x1
        else:
            self.x1 = x1
            self.x2 = x2
            self.x3 = x3

        # If they arent, cast all components to the same type
        if not (type(self.x1) == type(self.x2) and
                type(self.x1) == type(self.x3) and
                type(self.x2) == type(self.x3)):
                target_type = type(self.x1)
                self.x2 = target_type(self.x2)
                self.x3 = target_type(self.x3)

        if string_format is not None:
            self.string_format = string_format
        else:
            if type(self.x1) in [int]:
                self.string_format = "{:8d}"
            elif type(self.x1) in [float, np.float64]:
                self.string_format = "{:8.3f}"

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
            center_of_rotation = Vec3(0.0, 0.0, 0.0)
        x1offset = self.x1 - center_of_rotation.x1
        x2offset = self.x2 - center_of_rotation.x2
        self.x1prime = x1offset * cosd(theta) - x2offset * sind(theta) + center_of_rotation.x1
        self.x2prime = x2offset * cosd(theta) + x1offset * sind(theta) + center_of_rotation.x2
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
        return False not in np.isclose(
            [self.x1, self.x2, self.x3],
            [arg.x1, arg.x2, arg.x3]
        )

    def __hash__(self):
        return hash((self.x1, self.x2, self.x3))


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
    x = np.where(x <= -180., x + 360., x)
    x = np.where(x > 180., x - 360., x)
    return x


def wrap_360(x):
    """
    Shift the given values to within the range (0, 360].

    Args:
        x (numeric or np.array): Scalar value or np.array of values to shift.

    Returns:
        np.array: Shifted values.
    """
    x = np.where(x < 0., x + 360., x)
    x = np.where(x >= 360., x - 360., x)
    return x


class LogClass:
    """
    TODO
    """

    class __LogClass:
        """
        TODO
        """

        def __init__(self, param_dict):

            self.log_to_console = False
            self.console_level = 'WARNING'
            self.log_to_file = False
            self.file_level = 'WARNING'

            # TODO: what if it IS None?
            if param_dict is not None:
                for key in param_dict:
                    if key == 'console':
                        self.log_to_console = param_dict[key]['enable']
                        self.console_level = param_dict[key]['level']

                    if key == 'file':
                        self.log_to_file = param_dict[key]['enable']
                        self.file_level = param_dict[key]['level']

    instance = None

    def __init__(self, arg):
        if not LogClass.instance:
            LogClass.instance = self.__LogClass(arg)

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name, value):
        self.instance.__setattr__(name, value)


def setup_logger(name, logging_dict=None, floris=None):
    log_class = LogClass(logging_dict)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # level = 'WARNING'
    # level_styles = {'warning': {'color': 'red', 'bold': False}}
    fmt_console = '%(name)s %(levelname)s %(message)s'
    fmt_file = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    file_name = 'floris_{:%Y-%m-%d-%H_%M_%S}.log'.format(datetime.now())

    if log_class.log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_class.console_level)
        console_format = coloredlogs.ColoredFormatter(
            # level_styles=level_styles,
            fmt=fmt_console)
        console_handler.setFormatter(console_format)
        console_handler.addFilter(TracebackInfoFilter(clear=True))
        logger.addHandler(console_handler)

    if log_class.log_to_file:
        file_handler = logging.FileHandler(file_name)
        file_handler.setLevel(log_class.file_level)
        file_format = logging.Formatter(fmt_file)
        file_handler.setFormatter(file_format)
        file_handler.addFilter(TracebackInfoFilter(clear=False))
        logger.addHandler(file_handler)

    return logger


class TracebackInfoFilter(logging.Filter):
    """Clear or restore the exception on log records"""

    def __init__(self, clear=True):
        self.clear = clear

    def filter(self, record):
        if self.clear:
            record._stack_info_hidden, record.stack_info = \
                                                        record.stack_info, None
        elif hasattr(record, "_stack_info_hidden"):
            record.stack_info = record._stack_info_hidden
            del record._stack_info_hidden
        return True
