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

from ...utilities import cosd, sind, tand
from ...logging_manager import LoggerBase


class VelocityDeflection(LoggerBase):
    """
    This is the super-class for all wake deflection models. It includes
    implementations of functions that subclasses should use to perform
    secondary steering. See :cite:`bvd-King2019Controls`. for more details on
    how secondary steering is calculated.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: bvd-
    """

    def __init__(self, parameter_dictionary):
        """
        Stores the parameter dictionary for the wake deflection model.

        Args:
            parameter_dictionary (dict): Dictionary containing the wake
                deflection model parameters. See individual wake deflection
                models for details of specific key-value pairs.
        """

        self.model_string = None

        self.parameter_dictionary = parameter_dictionary

        # if 'use_secondary_steering' in self.parameter_dictionary:
        #     self.use_secondary_steering = \
        #         bool(self.parameter_dictionary["use_secondary_steering"])
        # else:
        #     self.logger.info('Using default option of applying gch-based ' + \
        #                 'secondary steering (use_secondary_steering=True)')
        #     self.use_secondary_steering = True

        # if 'eps_gain' in self.parameter_dictionary:
        #     self.eps_gain = bool(self.parameter_dictionary["eps_gain"])
        # else:
        #     # SOWFA SETTING (note this will be multiplied by D in function)
        #     self.eps_gain = 0.3
        #     self.logger.info(
        #         ('Using default option eps_gain: %.1f' % self.eps_gain))

    def _get_model_dict(self, default_dict):
        if self.model_string not in self.parameter_dictionary.keys():
            return_dict = default_dict
        else:
            user_dict = self.parameter_dictionary[self.model_string]
            # if default key is not in the user-supplied dict, then use the
            # default value
            for key in default_dict.keys():
                if key not in user_dict:
                    user_dict[key] = default_dict[key]
            # if user-supplied key is not in the default dict, then warn the
            # user that key: value pair was not used
            for key in user_dict:
                if key not in default_dict:
                    err_msg = (
                        "User supplied value {}, not in standard "
                        + "wake velocity model dictionary."
                    ).format(key)
                    self.logger.warning(err_msg, stack_info=True)
                    raise KeyError(err_msg)
            return_dict = user_dict
        return return_dict

    def calculate_effective_yaw_angle(
        self, x_locations, y_locations, z_locations, turbine, coord, flow_field
    ):
        """
        This method determines the effective yaw angle to be used when
        secondary steering is enabled. For more details on how the effective
        yaw angle is calculated, see :cite:`bvd-King2019Controls`.

        Args:
            x_locations (np.array): Streamwise locations in wake.
            y_locations (np.array): Spanwise locations in wake.
            z_locations (np.array): Vertical locations in wake.
            turbine (:py:class:`floris.simulation.turbine.Turbine`):
                Turbine object.
            coord (:py:obj:`floris.simulation.turbine_map.TurbineMap.coords`):
                Spatial coordinates of wind turbine.
            flow_field (:py:class:`floris.simulation.flow_field.FlowField`):
                Flow field object.

        Raises:
            ValueError: It appears that 'use_secondary_steering' is set
                to True and 'calculate_VW_velocities' is set to False.
                This configuration is not valid. Please set
                'use_secondary_steering' to True if you wish to use
                yaw-added recovery.

        Returns:
            float: The turbine yaw angle, including any effective yaw if
            secondary steering is enabled.
        """
        if self.use_secondary_steering:
            if not flow_field.wake.velocity_model.calculate_VW_velocities:
                err_msg = (
                    "It appears that 'use_secondary_steering' is set "
                    + "to True and 'calculate_VW_velocities' is set to False. "
                    + "This configuration is not valid. Please set "
                    + "'use_secondary_steering' to True if you wish to use "
                    + "yaw-added recovery."
                )
                self.logger.error(err_msg, stack_info=True)
                raise ValueError(err_msg)
            # turbine parameters
            Ct = turbine.Ct
            D = turbine.rotor_diameter
            HH = turbine.hub_height
            aI = turbine.aI
            TSR = turbine.tsr
            V = flow_field.v
            Uinf = np.mean(flow_field.wind_map.grid_wind_speed)

            eps = self.eps_gain * D  # Use set value
            xLocs = x_locations - coord.x1
            idx = np.where((np.abs(xLocs) < D / 4))

            yLocs = y_locations[idx] + 0.01 - coord.x2

            # location of top vortex
            zT = z_locations[idx] + 0.01 - (HH + D / 2)
            rT = yLocs ** 2 + zT ** 2

            # location of bottom vortex
            zB = z_locations[idx] + 0.01 - (HH - D / 2)
            rB = yLocs ** 2 + zB ** 2

            # wake rotation vortex
            zC = z_locations[idx] + 0.01 - (HH)
            rC = yLocs ** 2 + zC ** 2

            # find wake deflection from CRV
            test_gamma = np.linspace(-45, 45, 91)
            avg_V = np.mean(V[idx])
            minYaw = 10000
            target_yaw_ix = None
            for i in range(len(test_gamma)):

                # what yaw angle would have produced that same average spanwise velocity
                yaw = test_gamma[i]
                vel_top = (
                    Uinf
                    * ((HH + D / 2) / flow_field.specified_wind_height)
                    ** flow_field.wind_shear
                ) / Uinf
                vel_bottom = (
                    Uinf
                    * ((HH - D / 2) / flow_field.specified_wind_height)
                    ** flow_field.wind_shear
                ) / Uinf
                Gamma_top = (
                    (np.pi / 8) * D * vel_top * Uinf * Ct * sind(yaw) * cosd(yaw)
                )
                Gamma_bottom = (
                    -(np.pi / 8) * D * vel_bottom * Uinf * Ct * sind(yaw) * cosd(yaw)
                )
                Gamma_wake_rotation = (
                    0.25
                    * 2
                    * np.pi
                    * D
                    * (aI - aI ** 2)
                    * turbine.average_velocity
                    / TSR
                )
                Veff = (
                    (zT * Gamma_top) / (2 * np.pi * rT) * (1 - np.exp(-rT / (eps ** 2)))
                    + (zB * Gamma_bottom)
                    / (2 * np.pi * rB)
                    * (1 - np.exp(-rB / (eps ** 2)))
                    + (zC * Gamma_wake_rotation)
                    / (2 * np.pi * rC)
                    * (1 - np.exp(-rC / (eps ** 2)))
                )
                tmp = avg_V - np.mean(Veff)
                if np.abs(tmp) < minYaw:
                    minYaw = np.abs(tmp)
                    target_yaw_ix = i

            if target_yaw_ix is not None:
                yaw_effective = test_gamma[target_yaw_ix]
            else:
                err_msg = "No effective yaw angle is found. Set to 0."
                self.logger.warning(err_msg, stack_info=True)
                yaw_effective = 0.0

            return yaw_effective + turbine.yaw_angle

        else:
            return turbine.yaw_angle

    @property
    def use_secondary_steering(self):
        """
        Flag to use secondary steering on the wake deflection using methods
        developed in :cite:`bvd-King2019Controls`.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (bool): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._use_secondary_steering

    @use_secondary_steering.setter
    def use_secondary_steering(self, value):
        if type(value) is not bool:
            err_msg = (
                "Value of use_secondary_steering must be type "
                + "bool; {} given.".format(type(value))
            )
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._use_secondary_steering = value

    @property
    def eps_gain(self):
        """
        Tuning value for yaw added recovery on the wake velocity using methods
        developed in :cite:`bvd-King2019Controls`.

        TODO: Don't believe this needs to be defined here. Already defined in
        gaussian_model_model.py. Verify that it can be removed.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (bool): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._eps_gain

    @eps_gain.setter
    def eps_gain(self, value):
        if type(value) is not float:
            err_msg = "Value of eps_gain must be type " + "float; {} given.".format(
                type(value)
            )
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._eps_gain = value
