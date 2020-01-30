# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from ...utilities import cosd, sind, tand
from .base_velocity_deficit import VelocityDeficit
import numpy as np


class Gauss(VelocityDeficit):

    def __init__(self, parameter_dictionary):
        super().__init__(parameter_dictionary)
        self.model_string = "gauss"
        model_dictionary = self._get_model_dict()

        # wake expansion parameters
        self.ka = float(model_dictionary["ka"])
        self.kb = float(model_dictionary["kb"])

        # near wake parameters
        self.alpha = float(model_dictionary["alpha"])
        self.beta = float(model_dictionary["beta"])

    def function(self, x_locations, y_locations, z_locations, turbine, turbine_coord, deflection_field, flow_field):
        
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

        # initial velocity deficits
        uR = U_local * Ct / (2.0 * (1 - np.sqrt(1 - (Ct))))
        u0 = U_local * np.sqrt(1 - Ct)

        # initial Gaussian wake expansion
        sigma_z0 = D * 0.5 * np.sqrt(uR / (U_local + u0))
        sigma_y0 = sigma_z0 * cosd(yaw) * cosd(veer)

        # quantity that determines when the far wake starts
        x0 = D * (cosd(yaw) * (1 + np.sqrt(1 - Ct))) / (np.sqrt(2) \
            * (4 * self.alpha * TI + 2 * self.beta * (1 - np.sqrt(1 - Ct)))) \
            + turbine_coord.x1

        # wake expansion parameters
        ky = self.ka * TI + self.kb
        kz = self.ka * TI + self.kb

        # compute velocity deficit
        yR = y_locations - turbine_coord.x2
        xR = yR * tand(yaw) + turbine_coord.x1

        # velocity deficit in the near wake
        sigma_y = (((x0 - xR) - (x_locations - xR)) / (x0 - xR)) * 0.501 * \
            D * np.sqrt(Ct / 2.) + ((x_locations - xR) / (x0 - xR)) * sigma_y0
        sigma_z = (((x0 - xR) - (x_locations - xR)) / (x0 - xR)) * 0.501 * \
            D * np.sqrt(Ct / 2.) + ((x_locations - xR) / (x0 - xR)) * sigma_z0

        sigma_y[x_locations < xR] = 0.5 * D
        sigma_z[x_locations < xR] = 0.5 * D

        a = (cosd(veer)**2) / (2 * sigma_y**2) + \
            (sind(veer)**2) / (2 * sigma_z**2)
        b = -(sind(2 * veer)) / (4 * sigma_y**2) + \
            (sind(2 * veer)) / (4 * sigma_z**2)
        c = (sind(veer)**2) / (2 * sigma_y**2) + \
            (cosd(veer)**2) / (2 * sigma_z**2)
        totGauss = np.exp(-(a * ((y_locations - turbine_coord.x2) - delta)**2 \
                - 2 * b * ((y_locations - turbine_coord.x2) - delta) \
                * ((z_locations - HH)) + c * ((z_locations - HH))**2))

        velDef = (U_local * (1 - np.sqrt(1 - ((Ct * cosd(yaw)) \
                / (8.0 * sigma_y * sigma_z / D**2)))) * totGauss)
        velDef[x_locations < xR] = 0
        velDef[x_locations > x0] = 0

        # wake expansion in the lateral (y) and the vertical (z)
        sigma_y = ky * (x_locations - x0) + sigma_y0
        sigma_z = kz * (x_locations - x0) + sigma_z0

        sigma_y[x_locations < x0] = sigma_y0[x_locations < x0]
        sigma_z[x_locations < x0] = sigma_z0[x_locations < x0]

        # velocity deficit outside the near wake
        a = (cosd(veer)**2) / (2 * sigma_y**2) + \
            (sind(veer)**2) / (2 * sigma_z**2)
        b = -(sind(2 * veer)) / (4 * sigma_y**2) + \
            (sind(2 * veer)) / (4 * sigma_z**2)
        c = (sind(veer)**2) / (2 * sigma_y**2) + \
            (cosd(veer)**2) / (2 * sigma_z**2)
        totGauss = np.exp(-(a * ((y_locations - turbine_coord.x2) - delta)**2 \
                - 2 * b * ((y_locations - turbine_coord.x2) - delta) \
                * ((z_locations - HH)) + c * ((z_locations - HH))**2))

        # compute velocities in the far wake
        velDef1 = (U_local * (1 - np.sqrt(1 - ((Ct * cosd(yaw)) \
                / (8.0 * sigma_y * sigma_z / D**2)))) * totGauss)
        velDef1[x_locations < x0] = 0

        return np.sqrt(velDef**2 + velDef1**2), np.zeros(np.shape(velDef)), np.zeros(np.shape(velDef))
