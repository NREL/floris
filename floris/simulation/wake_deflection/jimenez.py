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
    Subclass of the
    :py:class:`floris.simulation.wake_deflection.VelocityDeflection`
    object class. Parameters required for Jimenez wake model:

     - ad: #TODO What is this parameter for?
     - kd: #TODO What is this parameter for?
     - bd: #TODO What is this parameter for?
    """
    
    default_parameters = {
        "kd": 0.05,
        "ad": 0.0,
        "bd": 0.0
    }

    def __init__(self, parameter_dictionary):
        """
        Instantiate Jimenez object and pass function paramter values.

        Args:
            parameter_dictionary (dict): input dictionary with the
                following key-value pairs:
                    {
                        "kd": 0.05,
                        "ad": 0.0,
                        "bd": 0.0
                    }
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
        This function defines the angle at which the wake deflects in
        relation to the yaw of the turbine. This is coded as defined in
        the Jimenez et. al. paper.

        Args:
            x_locations (np.array): streamwise locations in wake
            y_locations (np.array): spanwise locations in wake
            z_locations (np.array): vertical locations in wake
                (not used in Jimenez)
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
        yYaw_init = ( xi_init \
            * ( 15 * (2 * self.kd * x_locations \
            / turbine.rotor_diameter + 1)**4. + xi_init**2. ) \
            / ((30 * self.kd / turbine.rotor_diameter) \
            * (2 * self.kd * x_locations / turbine.rotor_diameter + 1)**5.)) \
            - (xi_init * turbine.rotor_diameter \
            * (15 + xi_init**2.) / (30 * self.kd))

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
        ... #TODO: Update docstring

        Args:
            kd (float, int): ... #TODO: Update docstring

        Returns:
            float: ... #TODO: Update docstring
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
        ... #TODO: Update docstring

        Args:
            ad (float, int): ... #TODO: Update docstring

        Returns:
            float: ... #TODO: Update docstring
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
        ... #TODO: Update docstring

        Args:
            bd (float, int): ... #TODO: Update docstring

        Returns:
            float: ... #TODO: Update docstring
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