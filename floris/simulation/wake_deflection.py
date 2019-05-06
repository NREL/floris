# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
from ..utilities import cosd, sind, tand


class WakeDeflection():
    """
    Base WakeDeflection object class. Subclasses are:

     - Jimenez
     - Gauss
     - Curl

    Each subclass has specific functional requirements. Refer to the
    each WakeDeflection subclass for further detail.
    """

    def __init__(self, parameter_dictionary):
        self.model_string = None

    def __str__(self):
        return self.model_string


class Jimenez(WakeDeflection):
    """
    Subclass of the
    :py:class:`floris.simulation.wake_deflection.WakeDeflection`
    object class. Parameters required for Jimenez wake model:

     - ad: #TODO What is this parameter for?
     - kd: #TODO What is this parameter for?
     - bd: #TODO What is this parameter for?
    """

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
        self.model_string = "jimenez"
        model_dictionary = parameter_dictionary[self.model_string]
        self.ad = float(model_dictionary["ad"])
        self.kd = float(model_dictionary["kd"])
        self.bd = float(model_dictionary["bd"])

    def function(self, x_locations, y_locations, turbine, coord, flow_field):
        """
        This function defines the angle at which the wake deflects in
        relation to the yaw of the turbine. This is coded as defined in
        the Jimenez et. al. paper.

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

        # angle of deflection
        xi_init = cosd(turbine.yaw_angle) * sind(
            turbine.yaw_angle) * turbine.Ct / 2.0

        x_locations = x_locations - coord.x1

        # yaw displacement
        yYaw_init = ( xi_init
            * ( 15 * (2 * self.kd * x_locations / turbine.rotor_diameter + 1)**4. + xi_init**2. )
            / ((30 * self.kd / turbine.rotor_diameter) * (2 * self.kd * x_locations / turbine.rotor_diameter + 1)**5.)) \
            - (xi_init * turbine.rotor_diameter * (15 + xi_init**2.) / (30 * self.kd))

        # corrected yaw displacement with lateral offset
        deflection = yYaw_init + self.ad + self.bd * x_locations

        x = np.unique(x_locations)
        for i in range(len(x)):
            tmp = np.max(deflection[x_locations == x[i]])
            deflection[x_locations == x[i]] = tmp

        return deflection


class Gauss(WakeDeflection):
    """
    Subclass of the
    :py:class:`floris.simulation.wake_deflection.WakeDeflection`
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
        model_dictionary = parameter_dictionary[self.model_string]
        self.ka = float(model_dictionary["ka"])
        self.kb = float(model_dictionary["kb"])
        self.ad = float(model_dictionary["ad"])
        self.bd = float(model_dictionary["bd"])
        self.alpha = float(model_dictionary["alpha"])
        self.beta = float(model_dictionary["beta"])
        self.deflection_multiplier = 1.0

    def function(self, x_locations, y_locations, turbine, coord, flow_field):
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
        wind_speed = flow_field.wind_speed  # free-stream velocity (m/s)
        TI_0 = flow_field.turbulence_intensity  # turbulence intensity (%/100)
        veer = flow_field.wind_veer  # veer (degrees)
        TI = TI_0

        # hard-coded model input data (goes in input file)
        ka = self.ka  # wake expansion parameter
        kb = self.kb  # wake expansion parameter
        alpha = self.alpha  # near wake parameter
        beta = self.beta  # near wake parameter
        ad = self.ad  # natural lateral deflection parameter
        bd = self.bd  # natural lateral deflection parameter

        # turbine parameters
        D = turbine.rotor_diameter
        yaw = -turbine.yaw_angle  # opposite sign convention in this model
        tilt = turbine.tilt_angle
        Ct = turbine.Ct

        # U_local = flow_field.wind_speed # just a placeholder for now, should be initialized with the flow_field
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
        )  # initial wake deflection; NOTE: use np.tan here since theta_c0 is radians

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


class Curl(WakeDeflection):
    """
    Subclass of the
    :py:class:`floris.simulation.wake_deflection.WakeDeflection`
    object. Parameters required for Curl wake model:

     - model_grid_resolution: #TODO What does this do?
    """

    def __init__(self, parameter_dictionary):
        """
        Instantiate Curl object and pass function paramter values.

        Args:
            parameter_dictionary (dict): input dictionary with the
                following key-value pair:
                    {
                        "model_grid_resolution": [
                                                    250,
                                                    100,
                                                    75
                                                ],
                    }
        """
        super().__init__(parameter_dictionary)
        self.model_string = "curl"

    def function(self, x_locations, y_locations, turbine, coord, flow_field):
        """
        This function will return the wake centerline predicted with
        the curled wake model. #TODO Eventually. This is coded as
        defined in the Martinez-Tossas et al. paper.
        """
        return np.zeros(np.shape(x_locations))
