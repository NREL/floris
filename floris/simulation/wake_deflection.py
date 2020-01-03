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

    def function(self, x_locations, y_locations, z_locations, turbine, coord, flow_field):
        """
        This function defines the angle at which the wake deflects in
        relation to the yaw of the turbine. This is coded as defined in
        the Jimenez et. al. paper.

        Args:
            x_locations (np.array): streamwise locations in wake
            y_locations (np.array): spanwise locations in wake
            z_locations (np.array): vertical locations in wake (not used in Jimenez)
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
        if 'dm' in model_dictionary:
            self.deflection_multiplier = float(model_dictionary["dm"])
        else:
            print('Using default gauss deflection multipler of 1.2')
            self.deflection_multiplier = 1.2

    def function(self, x_locations, y_locations, z_locations, turbine, coord, flow_field):
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
        wind_speed = flow_field.wind_map.grid_wind_speed  # free-stream velocity (m/s)
        veer = flow_field.wind_veer  # veer (degrees)
       
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
        yaw = -turbine.yaw_angle  # opposite sign convention in this model
        tilt = turbine.tilt_angle
        Ct = turbine.Ct

        # U_local = flow_field.wind_map.grid_wind_speed  #just a placeholder for now, should be initialized with the flow_field
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

    def function(self, x_locations, y_locations, z_locations, turbine, coord, flow_field):
        """
        This function will return the wake centerline predicted with
        the curled wake model. #TODO Eventually. This is coded as
        defined in the Martinez-Tossas et al. paper.
        """
        return np.zeros(np.shape(x_locations))

class GaussCurlHybrid(WakeDeflection):
    # TODO: update docstring

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
        self.model_string = "gauss_curl_hybrid"
        model_dictionary = parameter_dictionary[self.model_string]
        self.ka = float(model_dictionary["ka"])
        self.kb = float(model_dictionary["kb"])
        self.ad = float(model_dictionary["ad"])
        self.bd = float(model_dictionary["bd"])
        self.alpha = float(model_dictionary["alpha"])
        self.beta = float(model_dictionary["beta"])
        if 'dm' in model_dictionary:
            self.deflection_multiplier = float(model_dictionary["dm"])
        else:
            self.deflection_multiplier = 1.0
            # TODO: introduce logging
            print('Using default gauss deflection multipler of: %.1f' % self.deflection_multiplier)
            
        if 'use_ss' in model_dictionary:
            self.use_ss = bool(model_dictionary["use_ss"])
        else:
            # TODO: introduce logging
            print('Using default option of not applying gch-based secondary steering (use_ss=False)')
            self.use_ss = False

        if 'eps_gain' in model_dictionary:
            self.eps_gain = bool(model_dictionary["eps_gain"])
        else:
            self.eps_gain = 0.3 # SOWFA SETTING (note this will be multiplied by D in function)
            # TODO: introduce logging
            print('Using default option eps_gain: %.1f' % self.eps_gain)

    def function(self, x_locations, y_locations, z_locations, turbine, coord, flow_field):
        # TODO: update docstring

        # ==============================================================
        wind_speed = flow_field.wind_map.grid_wind_speed  # free-stream velocity (m/s)
        veer = flow_field.wind_veer  # veer (degrees)
       
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

        # GCH CODE
        if self.use_ss:
            # determine the effective yaw angle
            yaw_effective = self.effective_yaw(x_locations, y_locations, z_locations, coord, turbine, flow_field)
            yaw = -turbine.yaw_angle  - yaw_effective # opposite sign convention in this model
            print('Effective yaw angle = ', yaw_effective, turbine.yaw_angle)
        else:
            yaw = -turbine.yaw_angle

        tilt = turbine.tilt_angle
        Ct = turbine.Ct

        # U_local = flow_field.wind_map.grid_wind_speed  #just a placeholder for now, should be initialized with the flow_field
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

    def effective_yaw(self, x_locations, y_locations, z_locations, turbine_coord, turbine, flow_field):

        # turbine parameters
        Ct = turbine.Ct
        D = turbine.rotor_diameter
        HH = turbine.hub_height
        aI = turbine.aI
        TSR = turbine.tsr

        V = flow_field.v
        W = flow_field.w

        yLocs = y_locations - turbine_coord.x2
        zLocs = z_locations - (HH)

        # Use set value
        eps = self.eps_gain * D
        Uinf = np.mean(flow_field.wind_map.input_speed) # TODO Is this right?
        
        dist = np.sqrt(yLocs**2 + zLocs**2)
        xLocs = np.abs(x_locations - turbine_coord.x1)
        idx = np.where((dist < D/2) & (xLocs < D/4) & (np.abs(yLocs) > 0.1))

        Gamma = V[idx] * ((2 * np.pi) * (yLocs[idx] ** 2 + zLocs[idx] ** 2)) / (
                yLocs[idx] * (1 - np.exp(-(yLocs[idx] ** 2 + zLocs[idx] ** 2) / ((eps) ** 2))))
        Gamma_wake_rotation = 1.0 * 2 * np.pi * D * (aI - aI ** 2) * turbine.average_velocity / TSR
        Gamma0 = np.mean(np.abs(Gamma))

        test_gamma = np.linspace(-30, 30, 61)
        minYaw = 10000
        for i in range(len(test_gamma)):
            tmp1 = 8 * Gamma0 / (np.pi * flow_field.air_density * D * turbine.average_velocity * Ct)
            tmp = np.abs((sind(test_gamma[i]) * cosd(test_gamma[i]) ** 2) - tmp1)
            if tmp < minYaw:
                minYaw = tmp
                idx = i
        try:
            return test_gamma[idx]
        except:
            print('ERROR',idx)
            return 0.0
