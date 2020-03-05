# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from ...utilities import cosd, sind, tand
from .base_velocity_deficit import VelocityDeficit
import numpy as np
from scipy.special import gamma


class GaussianModel():
    """
    This is the first draft of what will hopefully become the new gaussian class 
    Currently it contains a direct port of the Bastankhah gaussian class from previous
    A direct implementation of the Blondel model
    And a new GM model where we merge features a bit more of the two to ensure consistency with previous far-wake results
    of the Gaussian model, while implementing the Blondel model's smooth near-wake

    TODO: This needs to be much more expanded and including full references

    [1] Abkar, M. and Porte-Agel, F. "Influence of atmospheric stability on
    wind-turbine wakes: A large-eddy simulation study." *Physics of
    Fluids*, 2015.

    [2] Bastankhah, M. and Porte-Agel, F. "A new analytical model for
    wind-turbine wakes." *Renewable Energy*, 2014.

    [3] Bastankhah, M. and Porte-Agel, F. "Experimental and theoretical
    study of wind turbine wakes in yawed conditions." *J. Fluid
    Mechanics*, 2016.

    [4] Niayifar, A. and Porte-Agel, F. "Analytical modeling of wind farms:
    A new approach for power prediction." *Energies*, 2016.

    [5] Dilip, D. and Porte-Agel, F. "Wind turbine wake mitigation through
    blade pitch offset." *Energies*, 2017.

    [6] Blondel, F. and Cathelain, M. "An alternative form of the
    super-Gaussian wind turbine wake model." *Wind Energy Science Disucssions*,
    2020.
    Notes to be written (merged)
    """    

    @staticmethod
    def mask_upstream_wake(y_locations, turbine_coord, yaw):
        yR = y_locations - turbine_coord.x2
        xR = yR * tand(yaw) + turbine_coord.x1
        return xR, yR

    @staticmethod
    def initial_velocity_deficits(U_local, Ct):
        uR = U_local * Ct / (2.0 * (1 - np.sqrt(1 - Ct)))
        u0 = U_local * np.sqrt(1 - Ct)
        return uR, u0

    @staticmethod
    def initial_wake_expansion(turbine, U_local, veer, uR, u0):
        yaw = -1 * turbine.yaw_angle 
        sigma_z0 = turbine.rotor_diameter * 0.5 * np.sqrt( uR / (U_local + u0) )
        sigma_y0 = sigma_z0 * cosd(yaw) * cosd(veer)
        return sigma_y0, sigma_z0

    @staticmethod
    def gaussian_function(U, C, r, n, sigma):
        return U * C * np.exp( -1 * r**n / (2 * sigma**2) )

    def __init__(self, parameter_dictionary):
        super().__init__(parameter_dictionary)
        self.model_string = "gauss_m"
        model_dictionary = self._get_model_dict()

        # wake expansion parameters
        self.ka = float(model_dictionary["ka"])
        self.kb = float(model_dictionary["kb"])

        # near wake / far wake boundary parameters
        self.alpha = float(model_dictionary["alpha"])
        self.beta = float(model_dictionary["beta"])

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

        # Current choices are g g2 b b2
        self.model_code = 'g' # or blondel


    def function(self, x_locations, y_locations, z_locations, turbine,
                 turbine_coord, deflection_field, flow_field):
        """
        First batch of code dedicated to sections common across codes
        """
        
        # veer (degrees)
        veer = flow_field.wind_veer

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

        # Calculate mask values to mask upstream wake
        yR = y_locations - turbine_coord.x2
        xR = yR * tand(yaw) + turbine_coord.x1

        # initial velocity deficits
        uR = U_local * Ct / (2.0 * (1 - np.sqrt(1 - Ct)))
        u0 = U_local * np.sqrt(1 - Ct)

        # initial Gaussian wake expansion
        sigma_z0 = D * 0.5 * np.sqrt( uR / (U_local + u0) )
        sigma_y0 = sigma_z0 * cosd(yaw) * cosd(veer)

        if self.model_code == 'gm':

            # MERGED Gaussian Code

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

            totGauss = np.exp( -r_tilde**n / (2 * sigma_tilde**2))

            a1 = 2**(2 / n - 1)
            a2 = 2**(4 / n - 2)
            C = a1 - np.sqrt(a2 - (n * Ct * cosd(yaw) / (16.0 * gamma(2/n) * np.sign(sigma_tilde) * np.abs(sigma_tilde)**(4/n) ) ) )

            # Compute wake velocity (Eq 1, pp 3 of ref. [1] in docstring)
            velDef1 = U_local * C * totGauss
            velDef1[x_locations < xR] = 0
            U = np.sqrt(velDef1**2)

        if self.model_code == 'g':

            # LEGACY GAUSS

            # quantity that determines when the far wake starts
            x0 = D * ( cosd(yaw) * (1 + np.sqrt(1 - Ct)) ) / ( np.sqrt(2) * ( 4 * self.alpha * TI + 2 * self.beta * ( 1 - np.sqrt(1 - Ct) ) ) ) + turbine_coord.x1

            # velocity deficit in the near wake
            sigma_y = (((x0 - xR) - (x_locations - xR)) / (x0 - xR)) * 0.501 * D * np.sqrt(Ct / 2.0) + ((x_locations - xR) / (x0 - xR)) * sigma_y0
            sigma_z = (((x0 - xR) - (x_locations - xR)) / (x0 - xR)) * 0.501 * D * np.sqrt(Ct / 2.0) + ((x_locations - xR) / (x0 - xR)) * sigma_z0
            sigma_y[x_locations < xR] = 0.5 * D
            sigma_z[x_locations < xR] = 0.5 * D

            a = cosd(veer)**2 / (2 * sigma_y**2) + sind(veer)**2 / (2 * sigma_z**2)
            b = -sind(2 * veer) / (4 * sigma_y**2) + sind(2 * veer) / (4 * sigma_z**2)
            c = sind(veer)**2 / (2 * sigma_y**2) + cosd(veer)**2 / (2 * sigma_z**2)
            totGauss = np.exp(-(a * ((y_locations - turbine_coord.x2) - delta)**2 - 2 * b * ((y_locations - turbine_coord.x2) - delta) * ((z_locations - HH)) + c * ((z_locations - HH))**2))

            velDef = (U_local * (1 - np.sqrt(1 - ((Ct * cosd(yaw)) / (8.0 * sigma_y * sigma_z / D**2)))) * totGauss)
            velDef[x_locations < xR] = 0
            velDef[x_locations > x0] = 0

            # wake expansion in the lateral (y) and the vertical (z)
            ky = self.ka * TI + self.kb  # wake expansion parameters
            kz = self.ka * TI + self.kb  # wake expansion parameters
            sigma_y = ky * (x_locations - x0) + sigma_y0
            sigma_z = kz * (x_locations - x0) + sigma_z0
            sigma_y[x_locations < x0] = sigma_y0[x_locations < x0]
            sigma_z[x_locations < x0] = sigma_z0[x_locations < x0]

            # velocity deficit outside the near wake
            a = cosd(veer)**2 / (2 * sigma_y**2) + sind(veer)**2 / (2 * sigma_z**2)
            b = -sind(2 * veer) / (4 * sigma_y**2) + sind(2 * veer) / (4 * sigma_z**2)
            c = sind(veer)**2 / (2 * sigma_y**2) + cosd(veer)**2 / (2 * sigma_z**2)
            totGauss = np.exp( -1 * (a * ((y_locations - turbine_coord.x2) - delta)**2 - 2 * b * ((y_locations - turbine_coord.x2) - delta) * ((z_locations - HH)) + c * ((z_locations - HH))**2))

            C = 1 - np.sqrt(1 - ( Ct * cosd(yaw) / (8.0 * sigma_y * sigma_z / D**2) ) ) 

            # compute velocities in the far wake
            velDef1 = U_local * C * totGauss
            velDef1[x_locations < x0] = 0
            U = np.sqrt(velDef**2 + velDef1**2)

        return U, np.zeros(np.shape(velDef1)), np.zeros(np.shape(velDef1))

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
        if type(value) is float:
            self._ka = value
        elif type(value) is int:
            self._ka = float(value)
        else:
            raise ValueError("Invalid value given for ka: {}".format(value))

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
        if type(value) is float:
            self._kb = value
        elif type(value) is int:
            self._kb = float(value)
        else:
            raise ValueError("Invalid value given for kb: {}".format(value))

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
        if type(value) is float:
            self._alpha = value
        elif type(value) is int:
            self._alpha = float(value)
        else:
            raise ValueError("Invalid value given for alpha: {}".format(value))

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
        if type(value) is float:
            self._beta = value
        elif type(value) is int:
            self._beta = float(value)
        else:
            raise ValueError("Invalid value given for beta: {}".format(value))