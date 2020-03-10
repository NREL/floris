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
from scipy.special import gamma
from ....utilities import cosd, sind, tand, setup_logger
from ..base_velocity_deficit import VelocityDeficit
from .gaussian_model_ish import GaussianModel


class MergeGauss(VelocityDeficit):
    default_parameters = {
        'ka': 0.38,
        'kb': 0.004,
        'alpha': 0.58,
        'beta': 0.077
    }

    def __init__(self, parameter_dictionary):
        super().__init__(parameter_dictionary)
        self.logger = setup_logger(name=__name__)

        self.model_string = "gauss_merge"
        model_dictionary = self._get_model_dict(__class__.default_parameters)

        # wake expansion parameters
        self.ka = model_dictionary["ka"]
        self.kb = model_dictionary["kb"]

        # near wake / far wake boundary parameters
        self.alpha = model_dictionary["alpha"]
        self.beta = model_dictionary["beta"]

    def function(self, x_locations, y_locations, z_locations, turbine, turbine_coord, deflection_field, flow_field):
        # added turbulence model
        TI = turbine.current_turbulence_intensity

        # turbine parameters
        D = turbine.rotor_diameter
        HH = turbine.hub_height
        yaw = -1 * turbine.yaw_angle  # opposite sign convention in this model
        Ct = turbine.Ct
        U_local = flow_field.u_initial

        # wake deflection
        delta = deflection_field

        xR, _ = GaussianModel.mask_upstream_wake(y_locations, turbine_coord, yaw)

        # Compute scaled variables (Eq 1, pp 3 of ref. [1] in docstring)
        x_tilde = (x_locations - turbine_coord.x1) / D
        r_tilde = np.sqrt( (y_locations - turbine_coord.x2 - delta)**2 + (z_locations - HH)**2, dtype=np.float128) / D

        beta = ( 1 + np.sqrt(1 - Ct * cosd(yaw)) )  /  (2 * ( 1 + np.sqrt(1 - Ct) ) )

        a_s = self.ka # Force equality to previous parameters to reduce new parameters
        b_s = self.kb # Force equality to previous parameters to reduce new parameters
        c_s = 0.5
        sigma_tilde = (a_s * TI + b_s) * x_tilde + c_s * np.sqrt(beta)

        x0 = D * ( cosd(yaw) * (1 + np.sqrt(1 - Ct))) / (np.sqrt(2) * (4 * self.alpha * TI + 2 * self.beta * (1 - np.sqrt(1 - Ct)))) + turbine_coord.x1
        sigma_tilde = sigma_tilde  - (a_s * TI + b_s) * x0/D

        a_f = 1.5 * 3.11
        b_f = 0.65 * -0.68
        c_f = 2.0
        n = a_f * np.exp(b_f * x_tilde) + c_f

        a1 = 2**(2 / n - 1)
        a2 = 2**(4 / n - 2)
        C = a1 - np.sqrt(a2 - (n * Ct * cosd(yaw) / (16.0 * gamma(2/n) * np.sign(sigma_tilde) * np.abs(sigma_tilde)**(4/n) ) ) )

        # Compute wake velocity (Eq 1, pp 3 of ref. [1] in docstring)
        velDef = GaussianModel.gaussian_function(U_local, C, r_tilde, n, sigma_tilde)
        velDef[x_locations < xR] = 0

        return velDef, np.zeros(np.shape(velDef)), np.zeros(np.shape(velDef))

    @property
    def ka(self):
        """
        Parameter used to determine the linear relationship between the 
            turbulence intensity and the width of the Gaussian wake shape.
        Args:
            ka (float, int): Gaussian wake model coefficient.
        Returns:
            float: Gaussian wake model coefficient.
        """
        return self._ka

    @ka.setter
    def ka(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for ka: {}, ' + \
                       'expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._ka = value
        if value != __class__.default_parameters['ka']:
            self.logger.info(
                ('Current value of ka, {0}, is not equal to tuned ' +
                'value of {1}.').format(
                    value, __class__.default_parameters['ka'])
                )

    @property
    def kb(self):
        """
        Parameter used to determine the linear relationship between the 
            turbulence intensity and the width of the Gaussian wake shape.
        Args:
            kb (float, int): Gaussian wake model coefficient.
        Returns:
            float: Gaussian wake model coefficient.
        """
        return self._kb

    @kb.setter
    def kb(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for kb: {}, ' + \
                       'expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._kb = value
        if value != __class__.default_parameters['kb']:
            self.logger.info(
                ('Current value of kb, {0}, is not equal to tuned ' +
                'value of {1}.').format(
                    value, __class__.default_parameters['kb'])
                )

    @property
    def alpha(self):
        """
        Parameter that determines the dependence of the downstream boundary
            between the near wake and far wake region on the turbulence
            intensity.
        Args:
            alpha (float, int): Gaussian wake model coefficient.
        Returns:
            float: Gaussian wake model coefficient.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for alpha: {}, ' + \
                       'expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._alpha = value
        if value != __class__.default_parameters['alpha']:
            self.logger.info(
                ('Current value of alpha, {0}, is not equal to tuned ' +
                'value of {1}.').format(
                    value, __class__.default_parameters['alpha'])
                )

    @property
    def beta(self):
        """
        Parameter that determines the dependence of the downstream boundary
            between the near wake and far wake region on the turbine's
            induction factor.
        Args:
            beta (float, int): Gaussian wake model coefficient.
        Returns:
            float: Gaussian wake model coefficient.
        """
        return self._beta

    @beta.setter
    def beta(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for beta: {}, ' + \
                       'expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._beta = value
        if value != __class__.default_parameters['beta']:
            self.logger.info(
                ('Current value of beta, {0}, is not equal to tuned ' +
                'value of {1}.').format(
                    value, __class__.default_parameters['beta'])
                )
