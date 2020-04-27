# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from ...utilities import cosd, sind, setup_logger
from .base_velocity_deflection import VelocityDeflection
import numpy as np


class Jimenez(VelocityDeflection):
    """
    Jiménez wake deflection model, dervied from
    :cite:`jdm-jimenez2010application`.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: jdm-
    """
    default_parameters = {
        "kd": 0.05,
        "ad": 0.0,
        "bd": 0.0
    }

    def __init__(self, parameter_dictionary):
        """
        Stores model parameters for use by methods.

        Args:
            parameter_dictionary (dict): Model-specific parameters.
                Default values are used when a parameter is not included
                in `parameter_dictionary`. Possible key-value pairs include:

                    -   **kd** (*float*): Parameter used to determine the skew
                        angle of the wake.
                    -   **ad** (*float*): Additional tuning parameter to modify
                        the wake deflection with a lateral offset.
                        Defaults to 0.
                    -   **bd** (*float*): Additional tuning parameter to modify
                        the wake deflection with a lateral offset.
                        Defaults to 0.

        """
        super().__init__(parameter_dictionary)
        self.logger = setup_logger(name=__name__)
        self.model_string = "jimenez"
        model_dictionary = self._get_model_dict(__class__.default_parameters)
        self.ad = float(model_dictionary["ad"])
        self.kd = float(model_dictionary["kd"])
        self.bd = float(model_dictionary["bd"])

    def function(self, x_locations, y_locations, z_locations, turbine, coord,
                 flow_field):
        """
        Calcualtes the deflection field of the wake in relation to the yaw of
        the turbine. This is coded as defined in [1].

        Args:
            x_locations (np.array): streamwise locations in wake
            y_locations (np.array): spanwise locations in wake
            z_locations (np.array): vertical locations in wake
                (not used in Jiménez)
            turbine (:py:class:`floris.simulation.turbine.Turbine`):
                Turbine object
            coord
                (:py:meth:`floris.simulation.turbine_map.TurbineMap.coords`):
                Spatial coordinates of wind turbine.
            flow_field
                (:py:class:`floris.simulation.flow_field.FlowField`):
                Flow field object.

        Returns:
            deflection (np.array): Deflected wake centerline.
        """

        # angle of deflection
        xi_init = cosd(turbine.yaw_angle) * sind(
            turbine.yaw_angle) * turbine.Ct / 2.0

        x_locations = x_locations - coord.x1

        # yaw displacement
        yYaw_init = ( xi_init * ( 15 * (2 * self.kd * x_locations / turbine.rotor_diameter + 1)**4. + xi_init**2. ) / ((30 * self.kd / turbine.rotor_diameter) 
            * (2 * self.kd * x_locations / turbine.rotor_diameter + 1)**5.)) - (xi_init * turbine.rotor_diameter * (15 + xi_init**2.) / (30 * self.kd))

        # corrected yaw displacement with lateral offset
        deflection = yYaw_init + self.ad + self.bd * x_locations

        x = np.unique(x_locations)
        for i in range(len(x)):
            tmp = np.max(deflection[x_locations == x[i]])
            deflection[x_locations == x[i]] = tmp

        return deflection

    @property
    def kd(self):
        """
        Parameter used to determine the skew angle of the wake.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._kd

    @kd.setter
    def kd(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for kd: {}, ' + \
                       'expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._kd = value
        if value != __class__.default_parameters['kd']:
            self.logger.info(
                ('Current value of kd, {0}, is not equal to tuned ' +
                'value of {1}.').format(
                    value, __class__.default_parameters['kd'])
                )

    @property
    def ad(self):
        """
        Parameter available for additional tuning of the wake deflection with a
        lateral offset.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._ad

    @ad.setter
    def ad(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for ad: {}, ' + \
                       'expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._ad = value
        if value != __class__.default_parameters['ad']:
            self.logger.info(
                ('Current value of ad, {0}, is not equal to tuned ' +
                'value of {1}.').format(
                    value, __class__.default_parameters['ad'])
                )

    @property
    def bd(self):
        """
        Parameter available for additional tuning of the wake deflection with a
        lateral offset.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._bd

    @bd.setter
    def bd(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for bd: {}, ' + \
                       'expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._bd = value
        if value != __class__.default_parameters['bd']:
            self.logger.info(
                ('Current value of bd, {0}, is not equal to tuned ' +
                'value of {1}.').format(
                    value, __class__.default_parameters['bd'])
                )