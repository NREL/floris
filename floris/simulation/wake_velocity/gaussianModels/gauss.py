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
from .gaussian_model_base import GaussianModel

class Gauss(GaussianModel):
    """
    The new Gauss model blends the previously implemented Gussian model based
    on [1-5] with the super-Gaussian model of [6].  The blending is meant to
    provide consistency with previous results in the far wake while improving
    prediction of the near wake.
    
    See :cite:`gvm-bastankhah2014new`, :cite:`gvm-abkar2015influence`,
    :cite:`gvm-bastankhah2016experimental`, :cite:`gvm-niayifar2016analytical`,
    :cite:`gvm-dilip2017wind`, :cite:`gvm-blondel2020alternative`, and
    :cite:`gvm-King2019Controls` for more information on Gaussian wake velocity
    deficit models.
    
    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: gvm-
    """
    default_parameters = {
        'ka': 0.38,
        'kb': 0.004,
        'alpha': 0.58,
        'beta': 0.077,
        'calculate_VW_velocities':True,
        'use_yaw_added_recovery':True,
        'yaw_recovery_alpha':0.03,
        'eps_gain':0.3
    }

    def __init__(self, parameter_dictionary):
        """
        Stores model parameters for use by methods.

        Args:
            parameter_dictionary (dict): Model-specific parameters.
                Default values are used when a parameter is not included
                in `parameter_dictionary`. Possible key-value pairs include:

                    -   **ka**: Parameter used to determine the linear
                        relationship between the turbulence intensity and the
                        width of the Gaussian wake shape.
                    -   **kb**: Parameter used to determine the linear
                        relationship between the turbulence intensity and the
                        width of the Gaussian wake shape.
                    -   **alpha**: Parameter that determines the dependence of
                        the downstream boundary between the near wake and far
                        wake region on the turbulence intensity.
                    -   **beta**: Parameter that determines the dependence of
                        the downstream boundary between the near wake and far
                        wake region on the turbine's induction factor.
                    -   **calculate_VW_velocities**: Flag to enable the
                        calculation of V- and W-component velocities using
                        methods developed in [7].
                    -   **use_yaw_added_recovery**: Flag to use yaw added
                        recovery on the wake velocity using methods developed
                        in [7].
                    -   **yaw_recovery_alpha**: Tuning value for yaw added
                        recovery on the wake velocity using methods developed
                        in [7].
                    -   **eps_gain**: Tuning value for calculating the V- and
                        W-component velocities using methods developed in [7].

        """
        super().__init__(parameter_dictionary)
        self.logger = setup_logger(name=__name__)

        self.model_string = "gauss"
        model_dictionary = self._get_model_dict(__class__.default_parameters)

        # wake expansion parameters
        self.ka = model_dictionary["ka"]
        self.kb = model_dictionary["kb"]

        # near wake / far wake boundary parameters
        self.alpha = model_dictionary["alpha"]
        self.beta = model_dictionary["beta"]

        # GCH Parameters
        self.calculate_VW_velocities = model_dictionary["calculate_VW_velocities"]
        self.use_yaw_added_recovery = model_dictionary["use_yaw_added_recovery"]
        self.yaw_recovery_alpha = model_dictionary["yaw_recovery_alpha"]
        self.eps_gain = model_dictionary["eps_gain"]

    def function(self, x_locations, y_locations, z_locations, turbine, turbine_coord, deflection_field, flow_field):
        """
        Using the blended Gaussian wake model, this method calculates and
        returns the wake velocity deficits, caused by the specified turbine,
        relative to the freestream velocities at the grid of points
        comprising the wind farm flow field.

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
                velocities for the U, V, and W components, aligned with the x, y,
                and z directions, respectively. The three arrays contain the
                velocity deficits at each grid point in the flow field.
        """
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

        # Over-ride the values less than xR, these go away anyway
        x_tilde[x_locations < xR] = 0 #np.mean(x_tilde[x_locations >= xR] )

        r_tilde = np.sqrt( (y_locations - turbine_coord.x2 - delta)**2 + (z_locations - HH)**2, dtype=np.float128) / D

        beta = ( 1 + np.sqrt(1 - Ct * cosd(yaw)) )  /  (2 * ( 1 + np.sqrt(1 - Ct) ) )

        a_s = self.ka # Force equality to previous parameters to reduce new parameters
        b_s = self.kb # Force equality to previous parameters to reduce new parameters
        c_s = 0.5
        
        x0 = D * ( cosd(yaw) * (1 + np.sqrt(1 - Ct))) / (np.sqrt(2) * (4 * self.alpha * TI + 2 * self.beta * (1 - np.sqrt(1 - Ct)))) # + turbine_coord.x1
        sigma_tilde = (a_s * TI + b_s) * (x_tilde - x0/D) + c_s * np.sqrt(beta)
        
        # If not subtracting x0 as above, but I think equivalent
        # sigma_tilde = (a_s * TI + b_s) * (x_tilde - 0) + c_s * np.sqrt(beta)
        # sigma_tilde = sigma_tilde  - (a_s * TI + b_s) * x0/D

        a_f = 1.5 * 3.11
        b_f = 0.65 * -0.68
        c_f = 2.0
        n = a_f * np.exp(b_f * x_tilde) + c_f

        a1 = 2**(2 / n - 1)
        a2 = 2**(4 / n - 2)
        
        # These two lines seem to be equivalent
        C = a1 - np.sqrt(a2 - (n * Ct * cosd(yaw) / (16.0 * gamma(2/n) * np.sign(sigma_tilde) * np.abs(sigma_tilde)**(4/n) ) ) )
        # C = a1 - np.sqrt(a2 - (n * Ct * cosd(yaw) / (16.0 * gamma(2/n) * sigma_tilde**(4/n) ) ) )

        # Compute wake velocity (Eq 1, pp 3 of ref. [1] in docstring)
        velDef = GaussianModel.gaussian_function(U_local, C, r_tilde, n, sigma_tilde)
        velDef[x_locations < xR] = 0

        return velDef, np.zeros(np.shape(velDef)), np.zeros(np.shape(velDef))

    @property
    def ka(self):
        """
        Parameter used to determine the linear relationship between the 
        turbulence intensity and the width of the Gaussian wake shape.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
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

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
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

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
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

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
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
