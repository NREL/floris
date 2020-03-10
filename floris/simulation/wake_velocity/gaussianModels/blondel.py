# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from ....utilities import cosd, sind, tand, setup_logger
from ..base_velocity_deficit import VelocityDeficit
import numpy as np
from scipy.special import gamma


class Blondel(VelocityDeficit):
    """
    Blondel is a velocity deficit subclass that contains objects...
    
    # TODO: update docstring
    [extended_summary]

    [1] Blondel, F. and Cathelain, M. "An alternative form of the
    super-Gaussian wind turbine wake model." *Wind Energy Science Disucssions*,
    2020.
    
    Args:
        VelocityDeficit ([type]): [description]
    
    Raises:
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]
    
    Returns:
        [type]: [description]
    """

    default_parameters = {
        "a_s": 0.3837,
        "b_s": 0.003678,
        "c_s": 0.2,
        "a_f": 3.11,
        "b_f": -0.68,
        "c_f": 2.41
    }

    def __init__(self, parameter_dictionary):
        super().__init__(parameter_dictionary)
        self.logger = setup_logger(name=__name__)

        self.model_string = "blondel"
        model_dictionary = self._get_model_dict(__class__.default_parameters)

        # wake expansion parameters
        # Table 2 of reference in docstring
        self.a_s = model_dictionary["a_s"]
        self.b_s = model_dictionary["b_s"]
        self.c_s = model_dictionary["c_s"]

        # fitted parameters for super-Gaussian order n
        # Table 3 of reference in docstring
        self.a_f = model_dictionary["a_f"]
        self.b_f = model_dictionary["b_f"]
        self.c_f = model_dictionary["c_f"]

        self.model_grid_resolution = None

    def function(self, x_locations, y_locations, z_locations, turbine,
                 turbine_coord, deflection_field, flow_field):

        # TODO: implement veer
        # Veer (degrees)
        veer = flow_field.wind_veer

        # Turbulence intensity for wake width calculation
        TI = turbine.current_turbulence_intensity

        # Turbine parameters
        D = turbine.rotor_diameter
        HH = turbine.hub_height
        yaw = -1 * turbine.yaw_angle  # opposite sign convention in this model
        Ct = turbine.Ct
        U_local = flow_field.u_initial

        # Wake deflection
        delta = deflection_field

        # Calculate mask values to mask upstream wake
        yR = y_locations - turbine_coord.x2
        xR = yR * tand(yaw) + turbine_coord.x1

        # Compute scaled variables (Eq 1, pp 3 of ref. [1] in docstring)
        x_tilde = (x_locations - turbine_coord.x1) / D
        r_tilde = np.sqrt( (y_locations - turbine_coord.x2 - delta)**2 \
                           + (z_locations - HH)**2, dtype=np.float128) / D

        # Calculate Beta (Eq 10, pp 5 of ref. [1] in docstring)
        beta = 0.5 * ((1 + np.sqrt(1 - Ct)) / np.sqrt(1 - Ct))

        # Calculate sigma_tilde (Eq 9, pp 5 of ref. [1] in docstring)
        sigma_tilde = (self.a_s * TI + self.b_s) * x_tilde + \
                       self.c_s * np.sqrt(beta)

        # Calculate n (Eq 13, pp 6 of ref. [1] in docstring)
        n = self.a_f * np.exp(self.b_f * x_tilde) + self.c_f

        # Calculate max vel def (Eq 5, pp 4 of ref. [1] in docstring)
        a1 = 2**(2 / n - 1)
        a2 = 2**(4 / n - 2)
        C = a1 - np.sqrt(a2 - ((n*Ct) * cosd(yaw) \
                / (16.0 * gamma(2/n) \
                * np.sign(sigma_tilde)*(np.abs(sigma_tilde)**(4/n)) )))

        # Compute wake velocity (Eq 1, pp 3 of ref. [1] in docstring)
        velDef1 = U_local * C * \
                    np.exp( (-1 * r_tilde**n) / (2 * sigma_tilde**2))
        velDef1[x_locations < xR] = 0

        return np.sqrt(velDef1**2), np.zeros(np.shape(velDef1)), \
                                    np.zeros(np.shape(velDef1))

    @property
    def a_s(self):
        """
        Constant coefficient used in calculation of wake expansion. See
            Eqn. 9 in "An alternative form of the super-Gaussian wind turbine
            wake model", Blondel et. al., Wind Energy Science Discussions, 2020.

        Args:
            a_s (float): Constant coefficient used in calculation of wake-added
                turbulence.

        Returns:
            float: Constant coefficient used in calculation of wake-added
                turbulence.
        """
        return self._a_s

    @a_s.setter
    def a_s(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for a_s: {}, ' + \
                       'expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._a_s = value
        if value != __class__.default_parameters['a_s']:
            self.logger.info(
                ('Current value of a_s, {0}, is not equal to tuned ' +
                'value of {1}.').format(
                    value, __class__.default_parameters['a_s'])
                )


    @property
    def b_s(self):
        """
        Constant coefficient used in calculation of wake expansion. See
            Eqn. 9 in "An alternative form of the super-Gaussian wind turbine
            wake model", Blondel et. al., Wind Energy Science Discussions, 2020.

        Args:
            b_s (float): Constant coefficient used in calculation of
                wake-added turbulence.

        Returns:
            float: Constant coefficient used in calculation of
                wake-added turbulence.
        """
        return self._b_s

    @b_s.setter
    def b_s(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for b_s: {}, ' + \
                       'expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._b_s = value
        if value != __class__.default_parameters['b_s']:
            self.logger.info(
                ('Current value of b_s, {0}, is not equal to tuned ' +
                'value of {1}.').format(
                    value, __class__.default_parameters['b_s'])
                )

    @property
    def c_s(self):
        """
        Linear constant used in calculation of wake expansion. See
            Eqn. 9 in "An alternative form of the super-Gaussian wind turbine
            wake model", Blondel et. al., Wind Energy Science Discussions, 2020.

        Args:
            c_s (float): Linear constant used in calculation of
                wake-added turbulence.

        Returns:
            float: Linear constant used in calculation of wake-added turbulence.
        """
        return self._c_s

    @c_s.setter
    def c_s(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for c_s: {}, ' + \
                       'expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._c_s = value
        if value != __class__.default_parameters['c_s']:
            self.logger.info(
                ('Current value of c_s, {0}, is not equal to tuned ' + \
                'value of {1}.').format(
                    value, __class__.default_parameters['c_s']
                )
            )

    @property
    def a_f(self):
        """
        Constant exponent coefficient used in calculation of the super-Gaussian
            order. See Eqn. 13 in "An alternative form of the super-Gaussian
            wind turbine wake model", Blondel et. al., Wind Energy Science
            Discussions, 2020.

        Args:
            a_f (float): Constant coefficient used in calculation the
                super-Gaussian order.

        Returns:
            float: Constant coefficient used in calculation the
                super-Gaussian order.
        """
        return self._a_f

    @a_f.setter
    def a_f(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for a_f: {}, ' + \
                       'expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._a_f = value
        if value != __class__.default_parameters['a_f']:
            self.logger.info(
                ('Current value of a_f, {0}, is not equal to tuned ' + \
                'value of {1}.').format(
                    value, __class__.default_parameters['a_f']
                )
            )

    @property
    def b_f(self):
        """
        Constant exponent coefficient used in calculation of the super-Gaussian
            order. See Eqn. 13 in "An alternative form of the super-Gaussian
            wind turbine wake model", Blondel et. al., Wind Energy Science
            Discussions, 2020.

        Args:
            b_f (float): Constant exponent coefficient used in calculation the
                super-Gaussian order.

        Returns:
            float: Constant exponent coefficient used in calculation the
                super-Gaussian order.
        """
        return self._b_f

    @b_f.setter
    def b_f(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for b_f: {}, ' + \
                       'expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._b_f = value
        if value != __class__.default_parameters['b_f']:
            self.logger.info(
                ('Current value of b_f, {0}, is not equal to tuned ' + \
                'value of {1}.').format(
                    value, __class__.default_parameters['b_f']
                )
            )

    @property
    def c_f(self):
        """
        Linear constant used in calculation of the super-Gaussian order. See
            Eqn. 13 in "An alternative form of the super-Gaussian wind turbine
            wake model", Blondel et. al., Wind Energy Science Discussions, 2020.

        Args:
            c_f (float): Linear constant used in calculation the
                super-Gaussian order.

        Returns:
            float: Linear constant used in calculation the super-Gaussian order.
        """
        return self._c_f

    @c_f.setter
    def c_f(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for c_f: {}, ' + \
                       'expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._c_f = value
        if value != __class__.default_parameters['c_f']:
            self.logger.info(
                ('Current value of c_f, {0}, is not equal to tuned ' + \
                'value of {1}.').format(
                    value, __class__.default_parameters['c_f']
                )
            )
