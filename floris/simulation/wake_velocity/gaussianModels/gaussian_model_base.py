# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from ....utilities import cosd, sind, tand
from ..base_velocity_deficit import VelocityDeficit
import numpy as np


class GaussianModel(VelocityDeficit):
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

    def __init__(self, parameter_dictionary):

        super().__init__(parameter_dictionary)


    def correction_steps(self, U_local, U, V, W, x_locations, y_locations,
                         turbine, turbine_coord):
        """
        TODO
        """
        if self.use_yaw_added_recovery:
            U = self.yaw_added_recovery_correction(U_local, U, W, \
                            x_locations, y_locations, turbine, turbine_coord)
        return U

    def calculate_VW(self, V, W, coord, turbine, flow_field, x_locations,
                     y_locations, z_locations):
        """
        # TODO
        """
        if self.calculate_VW_velocities:
            V, W = self.calc_VW(coord, turbine, flow_field, x_locations,
                                y_locations, z_locations)
        return V, W

    def yaw_added_recovery_correction(self, U_local, U, W, x_locations,
                                      y_locations, turbine, turbine_coord):
        """
        TODO
        """
        # compute the velocity without modification
        U1 = U_local - U

        # set dimensions
        xLocs = x_locations - turbine_coord.x1
        yLocs = y_locations - turbine_coord.x2
        # zLocs = z_locations
        D = turbine.rotor_diameter

        numerator = -1 * W * xLocs * np.abs(yLocs)
        denom = np.pi * (self.yaw_recovery_alpha * xLocs + D / 2)**2
        U2 = numerator / denom

        # add velocity modification from yaw (U2)
        # TODO: where would U2 be nan and should this be handled betteer?
        U_total = U1 + np.nan_to_num(U2)

        # turn it back into a deficit
        U = U_local - U_total

        # zero out anything before the turbine
        U[x_locations < turbine_coord.x1] = 0

        return U

    def calc_VW(self, coord, turbine, flow_field, x_locations, y_locations,
                z_locations):

        # turbine parameters
        D = turbine.rotor_diameter
        HH = turbine.hub_height
        yaw = turbine.yaw_angle
        Ct = turbine.Ct
        TSR = turbine.tsr
        aI = turbine.aI

        # flow parameters
        rho = flow_field.air_density

        # Update to wind map
        # Uinf = flow_field.wind_speed
        Uinf = np.mean(flow_field.wind_map.input_speed)  # TODO Is this right?

        # top point of the rotor
        dist_top = np.sqrt((coord.x1 - x_locations) ** 2 \
                            + ((coord.x2) - y_locations) ** 2 \
                            + (z_locations - (turbine.hub_height + D / 2)) ** 2)
        idx_top = np.where(dist_top == np.min(dist_top))

        # bottom point of the rotor
        dist_bottom = np.sqrt((coord.x1 - x_locations) ** 2 \
                            + ((coord.x2) - y_locations) ** 2 \
                            + (z_locations - (turbine.hub_height - D / 2)) ** 2)
        idx_bottom = np.where(dist_bottom == np.min(dist_bottom))

        if len(idx_top) > 1:
            idx_top = idx_top[0]
        if len(idx_bottom) > 1:
            idx_bottom = idx_bottom[0]

        scale = 1.0
        Gamma_top = scale * (np.pi / 8) * rho * D * turbine.average_velocity \
                    * Ct * sind(yaw) * cosd(yaw) ** 2
        Gamma_bottom = scale*(np.pi/8) * rho * D * turbine.average_velocity \
                       * Ct * sind(yaw) * cosd(yaw)**2
        Gamma_wake_rotation = 0.5 * 2 * np.pi * D * (aI - aI ** 2) \
                              * turbine.average_velocity / TSR

        # compute the spanwise and vertical velocities induced by yaw
        # Use set value
        eps = self.eps_gain * D

        # decay the vortices as they move downstream - using mixing length
        lmda = D / 8  #D/4 #D/4 #D/2
        kappa = 0.41
        lm = kappa * z_locations / (1 + kappa * z_locations / lmda)
        z = np.linspace(np.min(z_locations), np.max(z_locations), \
                        np.shape(flow_field.u_initial)[2])
        dudz_initial = np.gradient(flow_field.u_initial, z, axis=2)
        nu = lm**2 * np.abs(dudz_initial[0, :, :])

        # top vortex
        yLocs = y_locations + 0.01 - (coord.x2)
        zLocs = z_locations + 0.01 - (HH + D / 2)
        V1 = (((yLocs * Gamma_top) / (2 * np.pi * (yLocs**2 + zLocs**2))) \
                * (1 - np.exp(-(yLocs**2 + zLocs**2)/(eps**2))) ) * \
                eps**2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps**2)

        W1 = ((zLocs * Gamma_top) / (2 * np.pi * (yLocs**2 + zLocs**2))) \
               * (1 - np.exp(-(yLocs**2 + zLocs**2)/(eps**2))) * \
               eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        # bottom vortex
        yLocs = y_locations + 0.01 - (coord.x2)
        zLocs = z_locations + 0.01 - (HH - D / 2)
        V2 = (((yLocs * -Gamma_bottom) \
            / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))) \
            * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2)))) \
            * eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        W2 = ((zLocs * -Gamma_bottom) \
            / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))) \
            * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2))) \
            * eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        # top vortex - ground
        yLocs = y_locations + 0.01 - (coord.x2)
        zLocs = z_locations + 0.01 + (HH + D / 2)
        V3 = (((yLocs * -Gamma_top) \
            / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))) \
            * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2))) + 0.0) \
            * eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        W3 = ((zLocs * -Gamma_top) \
            / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))) \
            * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2))) \
            * eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        # bottom vortex - ground
        yLocs = y_locations + 0.01 - (coord.x2)
        zLocs = z_locations + 0.01 + (HH - D / 2)
        V4 = (((yLocs * Gamma_bottom) \
            / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))) \
            * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2))) + 0.0) \
            * eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        W4 = ((zLocs * Gamma_bottom) \
            / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))) \
            * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2))) \
            * eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        # wake rotation vortex
        yLocs = y_locations + 0.01 - coord.x2
        zLocs = z_locations + 0.01 - HH
        V5 = (((yLocs * Gamma_wake_rotation) \
            / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))) \
            * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2))) + 0.0) \
            * eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        W5 = ((zLocs * Gamma_wake_rotation) \
            / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))) \
            * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2))) \
            * eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        # wake rotation vortex - ground effect
        yLocs = y_locations + 0.01 - coord.x2
        zLocs = z_locations + 0.01 + HH
        V6 = (((yLocs * Gamma_wake_rotation) \
            / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))) \
            * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2))) + 0.0) \
            * eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        W6 = ((zLocs * Gamma_wake_rotation) \
            / (2 * np.pi * (yLocs ** 2 + zLocs ** 2))) \
            * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2))) \
            * eps ** 2 / (4 * nu * (x_locations - coord.x1) / Uinf + eps ** 2)

        # total spanwise velocity
        V = V1 + V2 + V3 + V4 + V5 + V6

        # total vertical velocity
        W = W1 + W2 + W3 + W4 + W5 + W6

        # compute velocity deficit
        # yR = y_locations - coord.x2
        # xR = yR * tand(yaw) + coord.x1
        V[x_locations < coord.x1 + 10] = 0.0
        W[x_locations < coord.x1 + 10] = 0.0

        # cut off in the spanwise direction
        V[np.abs(y_locations - coord.x2) > D] = 0.0
        W[np.abs(y_locations - coord.x2) > D] = 0.0

        return V, W

    @property
    def calculate_VW_velocities(self):
        return self._calculate_VW_velocities

    @calculate_VW_velocities.setter
    def calculate_VW_velocities(self, value):
        if type(value) is not bool:
            err_msg = "Value of calculate_VW_velocities must be type " + \
                      "float; {} given.".format(type(value))
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._calculate_VW_velocities = value

    @property
    def use_yaw_added_recovery(self):
        return self._use_yaw_added_recovery

    @use_yaw_added_recovery.setter
    def use_yaw_added_recovery(self, value):
        if type(value) is not bool:
            err_msg = "Value of use_yaw_added_recovery must be type " + \
                      "float; {} given.".format(type(value))
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._use_yaw_added_recovery = value

    @property
    def yaw_recovery_alpha(self):
        return self._yaw_recovery_alpha

    @yaw_recovery_alpha.setter
    def yaw_recovery_alpha(self, value):
        if type(value) is not float:
            err_msg = "Value of yaw_recovery_alpha must be type " + \
                      "float; {} given.".format(type(value))
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._yaw_recovery_alpha = value

    @property
    def eps_gain(self):
        return self._eps_gain

    @eps_gain.setter
    def eps_gain(self, value):
        if type(value) is not float:
            err_msg = "Value of eps_gain must be type " + \
                      "float; {} given.".format(type(value))
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._eps_gain = value


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
