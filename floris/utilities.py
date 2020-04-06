# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See read the https://floris.readthedocs.io for documentation

import numpy as np
from scipy.stats import norm
import coloredlogs
import logging
from datetime import datetime


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
            string_format: str (optional) -- The string format to use in the
                overloaded __str__ function.
        """
        if isinstance(x1, list):
            self.x1, self.x2, self.x3 = [x for x in x1]
        else:
            self.x1 = x1
            self.x2 = x2
            self.x3 = x3

        if not (type(self.x1) == type(self.x2) and type(self.x1) == type(
                self.x3) and type(self.x2) == type(self.x3)):
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


def calc_unc_pmfs(unc_options=None):
    # TODO: Is there a better place for this?
    """
    Calculates normally-distributed probability mass functions describing the 
    distribution of wind direction and yaw position deviations when wind direction 
    and/or yaw position uncertainty are included in power calculations.

    Args:
        unc_options (dictionary, optional): A dictionary containing values used 
                to create normally-distributed, zero-mean probability mass functions 
                describing the distribution of wind direction and yaw position 
                deviations when wind direction and/or yaw position uncertainty is 
                included. This argument is only used when **unc_pmfs** is None and 
                contains the following key-value pairs:

                -   **std_wd**: A float containing the standard deviation of the wind 
                        direction deviations from the original wind direction.
                -   **std_yaw**: A float containing the standard deviation of the yaw 
                        angle deviations from the original yaw angles.
                -   **pmf_res**: A float containing the resolution in degrees of the 
                        wind direction and yaw angle PMFs.
                -   **pdf_cutoff**: A float containing the cumulative distribution 
                    function value at which the tails of the PMFs are truncated. 

                Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw': 1.75, 
                'pmf_res': 1.0, 'pdf_cutoff': 0.995}.

    Returns:
        [dictionary]: A dictionary containing 
                probability mass functions describing the distribution of wind 
                direction and yaw position deviations when wind direction and/or 
                yaw position uncertainty is included in the power calculations. 
                Contains the following key-value pairs:  

                -   **wd_unc**: A numpy array containing wind direction deviations 
                    from the original wind direction. 
                -   **wd_unc_pmf**: A numpy array containing the probability of 
                    each wind direction deviation in **wd_unc** occuring. 
                -   **yaw_unc**: A numpy array containing yaw angle deviations 
                    from the original yaw angles. 
                -   **yaw_unc_pmf**: A numpy array containing the probability of 
                    each yaw angle deviation in **yaw_unc** occuring.

    """

    if unc_options is None:
        unc_options = {'std_wd': 4.95, 'std_yaw': 1.75, \
                    'pmf_res': 1.0, 'pdf_cutoff': 0.995}

    # create normally distributed wd and yaw uncertainty pmfs
    if unc_options['std_wd'] > 0:
        wd_bnd = int(np.ceil(norm.ppf(unc_options['pdf_cutoff'], \
                        scale=unc_options['std_wd'])/unc_options['pmf_res']))
        wd_unc = np.linspace(-1 * wd_bnd * unc_options['pmf_res'],
                             wd_bnd * unc_options['pmf_res'], 2 * wd_bnd + 1)
        wd_unc_pmf = norm.pdf(wd_unc, scale=unc_options['std_wd'])
        wd_unc_pmf = wd_unc_pmf / np.sum(wd_unc_pmf)  # normalize so sum = 1.0
    else:
        wd_unc = np.zeros(1)
        wd_unc_pmf = np.ones(1)

    if unc_options['std_yaw'] > 0:
        yaw_bnd = int(np.ceil(norm.ppf(unc_options['pdf_cutoff'], \
                        scale=unc_options['std_yaw'])/unc_options['pmf_res']))
        yaw_unc = np.linspace(-1 * yaw_bnd * unc_options['pmf_res'],
                              yaw_bnd * unc_options['pmf_res'],
                              2 * yaw_bnd + 1)
        yaw_unc_pmf = norm.pdf(yaw_unc, scale=unc_options['std_yaw'])
        yaw_unc_pmf = yaw_unc_pmf / np.sum(
            yaw_unc_pmf)  # normalize so sum = 1.0
    else:
        yaw_unc = np.zeros(1)
        yaw_unc_pmf = np.ones(1)

    return {'wd_unc': wd_unc, 'wd_unc_pmf': wd_unc_pmf, \
                'yaw_unc': yaw_unc, 'yaw_unc_pmf': yaw_unc_pmf}


class LogClass:
    class __LogClass:
        def __init__(self, param_dict):

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
