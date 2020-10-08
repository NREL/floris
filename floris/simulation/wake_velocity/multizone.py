# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np

from ...utilities import cosd
from .base_velocity_deficit import VelocityDeficit


class MultiZone(VelocityDeficit):
    """
    The MultiZone model computes the wake velocity deficit based
    on the original multi-zone FLORIS model. See
    :cite:`mvm-gebraad2014data,mvm-gebraad2016wind` for more details.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: mvm-
    """

    default_parameters = {
        "me": [-0.5, 0.3, 1.0],
        "we": 0.05,
        "aU": 12.0,
        "bU": 1.3,
        "mU": [0.5, 1.0, 5.5],
    }

    def __init__(self, parameter_dictionary):
        """
        Stores model parameters for use by methods.

        Args:
            parameter_dictionary (dict): Model-specific parameters.
                Default values are used when a parameter is not included
                in `parameter_dictionary`. Possible key-value pairs include:

                -   **me** (*list*): A list of three floats that help determine
                    the slope of the diameters of the three wake zones
                    (near wake, far wake, mixing zone) as a function of
                    downstream distance.
                -   **we** (*float*): Scaling parameter used to adjust the wake
                    expansion, helping to determine the slope of the diameters
                    of the three wake zones as a function of downstream
                    distance, as well as the recovery of the velocity deficits
                    in the wake as a function of downstream distance.
                -   **aU** (*float*): A float that is a parameter used to
                    determine the dependence of the wake velocity deficit decay
                    rate on the rotor yaw angle.
                -   **bU** (*float*): A float that is another parameter used to
                    determine the dependence of the wake velocity deficit decay
                    rate on the rotor yaw angle.
                -   **mU** (*list*): A list of three floats that are parameters
                    used to determine the dependence of the wake velocity
                    deficit decay rate for each of the three wake zones on the
                    rotor yaw angle.
        """
        super().__init__(parameter_dictionary)
        self.model_string = "multizone"
        model_dictionary = self._get_model_dict(__class__.default_parameters)
        self.me = [n for n in model_dictionary["me"]]
        self.we = model_dictionary["we"]
        self.aU = model_dictionary["aU"]
        self.bU = model_dictionary["bU"]
        self.mU = [n for n in model_dictionary["mU"]]

    def function(
        self,
        x_locations,
        y_locations,
        z_locations,
        turbine,
        turbine_coord,
        deflection_field,
        flow_field,
    ):
        """
        Using the original FLORIS multi-zone wake model, this method
        calculates and returns the wake velocity deficits, caused by
        the specified turbine, relative to the freestream velocities at
        the grid of points comprising the wind farm flow field.

        Args:
            x_locations (np.array): An array of floats that contains the
                streamwise direction grid coordinates of the flow field
                domain (m).
            y_locations (np.array): An array of floats that contains the grid
                coordinates of the flow field domain in the direction normal to
                x and parallel to the ground (m).
            z_locations (np.array): An array of floats that contains the grid
                coordinates of the flow field domain in the vertical
                direction (m).
            turbine (:py:obj:`floris.simulation.turbine`): Object that
                represents the turbine creating the wake.
            turbine_coord (:py:obj:`floris.utilities.Vec3`): Object containing
                the coordinate of the turbine creating the wake (m).
            deflection_field (np.array): An array of floats that contains the
                amount of wake deflection in meters in the y direction at each
                grid point of the flow field.
            flow_field (:py:class:`floris.simulation.flow_field`): Object
                containing the flow field information for the wind farm.

        Returns:
            np.array, np.array, np.array:
                Three arrays of floats that contain the wake velocity
                deficit in m/s created by the turbine relative to the freestream
                velocities for the U, V, and W components, aligned with the
                x, y, and z directions, respectively. The three arrays contain
                the velocity deficits at each grid point in the flow field.
        """

        mu = self.mU / cosd(self.aU + self.bU * turbine.yaw_angle)

        # distance from wake centerline
        rY = abs(y_locations - (turbine_coord.x2 + deflection_field))
        # rZ = abs(z_locations - (turbine.hub_height))
        dx = x_locations - turbine_coord.x1

        # wake zone diameters
        nearwake = turbine.rotor_radius + self.we * self.me[0] * dx
        farwake = turbine.rotor_radius + self.we * self.me[1] * dx
        mixing = turbine.rotor_radius + self.we * self.me[2] * dx

        # initialize the wake field
        c = np.zeros(x_locations.shape)

        # near wake zone
        mask = rY <= nearwake
        c += (
            mask
            * (
                turbine.rotor_diameter
                / (turbine.rotor_diameter + 2 * self.we * mu[0] * dx)
            )
            ** 2
        )
        # mask = rZ <= nearwake
        # c += mask * (radius / (radius + we * mu[0] * dx))**2

        # far wake zone
        # ^ is XOR, x^y:
        #   Each bit of the output is the same as the corresponding bit in x
        #   if that bit in y is 0, and it's the complement of the bit in x
        #   if that bit in y is 1.
        # The resulting mask is all the points in far wake zone that are not
        # in the near wake zone
        mask = (rY <= farwake) ^ (rY <= nearwake)
        c += (
            mask
            * (
                turbine.rotor_diameter
                / (turbine.rotor_diameter + 2 * self.we * mu[1] * dx)
            )
            ** 2
        )
        # mask = (rZ <= farwake) ^ (rZ <= nearwake)
        # c += mask * (radius / (radius + we * mu[1] * dx))**2

        # mixing zone
        # | is OR, x|y:
        #   Each bit of the output is 0 if the corresponding bit of x AND
        #   of y is 0, otherwise it's 1.
        # The resulting mask is all the points in mixing zone that are not
        # in the far wake zone and not in  near wake zone
        mask = (rY <= mixing) ^ ((rY <= farwake) | (rY <= nearwake))
        c += (
            mask
            * (
                turbine.rotor_diameter
                / (turbine.rotor_diameter + 2 * self.we * mu[2] * dx)
            )
            ** 2
        )
        # mask = (rZ <= mixing) ^ ((rZ <= farwake) | (rZ <= nearwake))
        # c += mask * (radius / (radius + we * mu[2] * dx))**2

        # filter points upstream
        c[x_locations - turbine_coord.x1 < 0] = 0

        return (
            2 * turbine.aI * c * flow_field.wind_map.grid_wind_speed,
            np.zeros(np.shape(c)),
            np.zeros(np.shape(c)),
        )

    @property
    def me(self):
        """
        A list of three floats that help determine the slope of the diameters
        of the three wake zones (near wake, far wake, mixing zone) as a
        function of downstream distance.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (list): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._me

    @me.setter
    def me(self, value):
        # if type(value) is list and len(value) == 3 and \
        #                     all(type(val) is float for val in value) is True:
        #     self._me = value
        # elif type(value) is list and len(value) == 3 and \
        #                     all(type(val) is float for val in value) is False:
        #     self._me = [float(val) for val in value]
        # else:
        #     raise ValueError("Invalid value given for me: {}".format(value))

        if (
            type(value) is not list
            or len(value) != 3
            or not all(type(val) is float for val in value)
        ):
            err_msg = (
                "Invalid value type given for me: {}, " + "expected list of length 3."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._me = value
        if value != __class__.default_parameters["me"]:
            self.logger.info(
                (
                    "Current value of me, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["me"])
            )

    @property
    def we(self):
        """
        Scaling parameter used to adjust the wake expansion, helping to
        determine the slope of the diameters of the three wake zones as a
        function of downstream distance, as well as the recovery of the
        velocity deficits in the wake as a function of downstream distance.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._we

    @we.setter
    def we(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for we: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._we = value
        if value != __class__.default_parameters["we"]:
            self.logger.info(
                (
                    "Current value of we, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["we"])
            )

    @property
    def aU(self):
        """
        Parameter used to determine the dependence of the wake velocity deficit
        decay rate on the rotor yaw angle.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._aU

    @aU.setter
    def aU(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for aU: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._aU = value
        if value != __class__.default_parameters["aU"]:
            self.logger.info(
                (
                    "Current value of aU, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["aU"])
            )

    @property
    def bU(self):
        """
        Parameter used to determine the dependence of the wake velocity deficit
        decay rate on the rotor yaw angle.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._bU

    @bU.setter
    def bU(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for bU: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._bU = value
        if value != __class__.default_parameters["bU"]:
            self.logger.info(
                (
                    "Current value of bU, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["bU"])
            )

    @property
    def mU(self):
        """
        A list of three floats that are parameters used to determine the
        dependence of the wake velocity deficit decay rate for each of the
        three wake zones on the rotor yaw angle.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (list): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._mU

    @mU.setter
    def mU(self, value):
        # if type(value) is list and len(value) == 3 and \
        #                     all(type(val) is float for val in value) is True:
        #     self._mU = value
        # elif type(value) is list and len(value) == 3 and \
        #                     all(type(val) is float for val in value) is False:
        #     self._mU = [float(val) for val in value]
        # else:
        #     raise ValueError("Invalid value given for mU: {}".format(value))

        if (
            type(value) is not list
            or len(value) != 3
            or not all(type(val) is float for val in value)
        ):
            err_msg = (
                "Invalid value type given for mU: {}, " + "expected list of length 3."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._mU = value
        if value != __class__.default_parameters["mU"]:
            self.logger.info(
                (
                    "Current value of mU, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["mU"])
            )
