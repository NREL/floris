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
from ...utilities import cosd, sind, tand, setup_logger


class VelocityDeficit():
    """
    VelocityDeficit is the base class of the different velocity deficit model 
    classes.

    An instantiated VelocityDeficit object will import parameters used to 
    calculate wake-added turbulence intensity from an upstream turbine, 
    using the approach of Crespo, A. and Herna, J., "Turbulence 
    characteristics in wind-turbine wakes." *J. Wind Eng Ind Aerodyn*. 
    1996.

    Args:
        parameter_dictionary: A dictionary as generated from the 
            input_reader; it should have the following key-value pairs:

            -   **turbulence_intensity**: A dictionary containing the 
                following key-value pairs:

                -   **initial**: A float that is the initial ambient 
                    turbulence intensity, expressed as a decimal 
                    fraction.
                -   **constant**: A float that is the constant used to 
                    scale the wake-added turbulence intensity.
                -   **ai**: A float that is the axial induction factor 
                    exponent used in in the calculation of wake-added 
                    turbulence.
                -   **downstream**: A float that is the exponent 
                    applied to the distance downtream of an upstream 
                    turbine normalized by the rotor diameter used in 
                    the calculation of wake-added turbulence.

    Returns:
        An instantiated VelocityDeficit object.
    """

    def __init__(self, parameter_dictionary):
        self.logger = setup_logger(name=__name__)

        self.requires_resolution = False
        self.model_string = None
        self.model_grid_resolution = None
        self.parameter_dictionary = parameter_dictionary

        if 'calculate_VW_velocities' in self.parameter_dictionary:
            self.calculate_VW_velocities = \
                bool(self.parameter_dictionary["calculate_VW_velocities"])
        else:
            self.logger.info('Using default option of not calculating V and W ' + \
                'velocity components (calculate_VW_velocities=False)')
            self.calculate_VW_velocities = False

        if 'use_yaw_added_recovery' in self.parameter_dictionary:
            # if set to True, self.calculate_VW_velocities also is set to True
            self.use_yaw_added_recovery = \
                bool(self.parameter_dictionary["use_yaw_added_recovery"])
        else:
            self.logger.info('Using default option of not applying added ' + \
                        'yaw-added recovery (use_yaw_added_recovery=True)')
            self.use_yaw_added_recovery = False

        if 'yaw_recovery_alpha' in self.parameter_dictionary:
            self.yaw_recovery_alpha = \
                bool(self.parameter_dictionary["yaw_recovery_alpha"])
        else:
            self.yaw_recovery_alpha = 0.03
            self.logger.info('Using default option yaw_recovery_alpha: %.2f' \
                        % self.yaw_recovery_alpha)

        if 'eps_gain' in self.parameter_dictionary:
            self.eps_gain = bool(self.parameter_dictionary["eps_gain"])
        else:
            self.eps_gain = 0.3  # SOWFA SETTING (note this will be multiplied
            # by D in function)
            self.logger.info(
                ('Using default option eps_gain: %.1f' % self.eps_gain))

    def _get_model_dict(self, default_dict):
        if self.model_string not in self.parameter_dictionary.keys():
            return_dict = default_dict
        else:
            user_dict = self.parameter_dictionary[self.model_string]
            # if default key is not in the user-supplied dict, then use the
            # default value
            for key in default_dict.keys():
                if key not in user_dict:
                    user_dict[key] = default_dict[key]
            # if user-supplied key is not in the default dict, then warn the
            # user that key: value pair was not used
            for key in user_dict:
                if key not in default_dict:
                    err_msg = ('User supplied value {}, not in standard ' + \
                        'wake velocity model dictionary.').format(key)
                    self.logger.warning(err_msg, stack_info=True)
                    raise KeyError(err_msg)
            return_dict = user_dict
        return return_dict

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
