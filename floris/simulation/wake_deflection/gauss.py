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
from .base_velocity_deflection import VelocityDeflection
import numpy as np


class Gauss(VelocityDeflection):
    """
    Subclass of the
    :py:class:`floris.simulation.wake_deflection.VelocityDeflection`
    object. Parameters required for Gauss wake model:

     - ka: #TODO What is this parameter for?
     - kb: #TODO What is this parameter for?
     - alpha: #TODO What is this parameter for?
     - beta: #TODO What is this parameter for?
     - ad: #TODO What is this parameter for?
     - bd: #TODO What is this parameter for?
    """

    def __init__(self, parameter_dictionary):
        """
        Instantiate Gauss object and pass function paramter values.

        Args:
            parameter_dictionary (dict): input dictionary with the
                following key-value pairs:
                    {
                        "ka": 0.3,
                        "kb": 0.004,
                        "alpha": 0.58,
                        "beta": 0.077,
                        "ad": 0.0,
                        "bd": 0.0
                    }
        """
        super().__init__(parameter_dictionary)
        self.model_string = "gauss"
        model_dictionary = self._get_model_dict()
        self.ka = float(model_dictionary["ka"])
        self.kb = float(model_dictionary["kb"])
        self.ad = float(model_dictionary["ad"])
        self.bd = float(model_dictionary["bd"])
        self.alpha = float(model_dictionary["alpha"])
        self.beta = float(model_dictionary["beta"])
        if 'dm' in model_dictionary:
            self.deflection_multiplier = float(model_dictionary["dm"])
        else:
            self.deflection_multiplier = 1.2
            # TODO: logging
            print('Using default gauss deflection multipler of: %.1f' %
                  self.deflection_multiplier)

    def function(self, x_locations, y_locations, z_locations, turbine, coord,
                 flow_field):
        """
        This function defines the angle at which the wake deflects in
        relation to the yaw of the turbine. This is coded as defined in
        the Bastankah and Porte_Agel paper.

        Args:
            x_locations (np.array): streamwise locations in wake
            y_locations (np.array): spanwise locations in wake
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
        # ==============================================================
        # free-stream velocity (m/s)
        wind_speed = flow_field.wind_map.grid_wind_speed
        # veer (degrees)
        veer = flow_field.wind_veer

        # added turbulence model
        TI = turbine.current_turbulence_intensity
        # TI = flow_field.wind_map.grid_turbulence_intensity
        # hard-coded model input data (goes in input file)
        ka = self.ka  # wake expansion parameter
        kb = self.kb  # wake expansion parameter
        alpha = self.alpha  # near wake parameter
        beta = self.beta  # near wake parameter
        ad = self.ad  # natural lateral deflection parameter
        bd = self.bd  # natural lateral deflection parameter

        # turbine parameters
        D = turbine.rotor_diameter
        yaw = -1 * self.calculate_effective_yaw_angle(x_locations, y_locations,
                            z_locations, turbine, coord, flow_field)
                            # opposite sign convention in this model
        tilt = turbine.tilt_angle
        Ct = turbine.Ct

        # U_local = flow_field.wind_map.grid_wind_speed  
        # just a placeholder for now, should be initialized with the flow_field
        U_local = flow_field.u_initial

        # initial velocity deficits
        uR = U_local * Ct * cosd(tilt) * cosd(yaw) / (
            2. * (1 - np.sqrt(1 - (Ct * cosd(tilt) * cosd(yaw)))))
        u0 = U_local * np.sqrt(1 - Ct)

        # length of near wake
        x0 = D * (cosd(yaw) * (1 + np.sqrt(1 - Ct * cosd(yaw)))) / (
            np.sqrt(2) * (4 * alpha * TI + 2 * beta *
                          (1 - np.sqrt(1 - Ct)))) + coord.x1

        # wake expansion parameters
        ky = ka * TI + kb
        kz = ka * TI + kb

        C0 = 1 - u0 / wind_speed
        M0 = C0 * (2 - C0)
        E0 = C0**2 - 3 * np.exp(1. / 12.) * C0 + 3 * np.exp(1. / 3.)

        # initial Gaussian wake expansion
        sigma_z0 = D * 0.5 * np.sqrt(uR / (U_local + u0))
        sigma_y0 = sigma_z0 * cosd(yaw) * cosd(veer)

        yR = y_locations - coord.x2
        xR = yR * tand(yaw) + coord.x1

        # yaw parameters (skew angle and distance from centerline)
        theta_c0 = self.deflection_multiplier * (
            0.3 * np.radians(yaw) / cosd(yaw)) * (
                1 - np.sqrt(1 - Ct * cosd(yaw)))  # skew angle in radians
        delta0 = np.tan(theta_c0) * (
            x0 - coord.x1
        )  # initial wake deflection;
        # NOTE: use np.tan here since theta_c0 is radians

        # deflection in the near wake
        delta_near_wake = ((x_locations - xR) /
                           (x0 - xR)) * delta0 + (ad + bd *
                                                  (x_locations - coord.x1))
        delta_near_wake[x_locations < xR] = 0.0
        delta_near_wake[x_locations > x0] = 0.0

        # deflection in the far wake
        sigma_y = ky * (x_locations - x0) + sigma_y0
        sigma_z = kz * (x_locations - x0) + sigma_z0
        sigma_y[x_locations < x0] = sigma_y0[x_locations < x0]
        sigma_z[x_locations < x0] = sigma_z0[x_locations < x0]

        ln_deltaNum = (1.6 + np.sqrt(M0)) * (
            1.6 * np.sqrt(sigma_y * sigma_z /
                          (sigma_y0 * sigma_z0)) - np.sqrt(M0))
        ln_deltaDen = (1.6 - np.sqrt(M0)) * (
            1.6 * np.sqrt(sigma_y * sigma_z /
                          (sigma_y0 * sigma_z0)) + np.sqrt(M0))
        delta_far_wake = delta0 + (theta_c0 * E0 / 5.2) * np.sqrt(
            sigma_y0 * sigma_z0 /
            (ky * kz * M0)) * np.log(ln_deltaNum / ln_deltaDen) + (
                ad + bd * (x_locations - coord.x1))
        delta_far_wake[x_locations <= x0] = 0.0

        deflection = delta_near_wake + delta_far_wake

        return deflection

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

    @property
    def ad(self):
        """
        ... #TODO: update docstring

        Args:
            ad (float, int): ... #TODO: update docstring

        Returns:
            float: ... #TODO: update docstring
        """
        return self._ad

    @ad.setter
    def ad(self, value):
        if type(value) is float:
            self._ad = value
        elif type(value) is int:
            self._ad = float(value)
        else:
            raise ValueError("Invalid value given for ad: {}".format(value))

    @property
    def bd(self):
        """
        ... #TODO: update docstring

        Args:
            bd (float, int): ... #TODO: update docstring

        Returns:
            float: ... #TODO: update docstring
        """
        return self._bd

    @bd.setter
    def bd(self, value):
        if type(value) is float:
            self._bd = value
        elif type(value) is int:
            self._bd = float(value)
        else:
            raise ValueError("Invalid value given for bd: {}".format(value))
