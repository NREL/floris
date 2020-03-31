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


class VelocityDeflection():
    """
    Base VelocityDeflection object class. Subclasses are:

    Each subclass has specific functional requirements. Refer to the
    each VelocityDeflection subclass for further detail.
    """

    def __init__(self, parameter_dictionary):
        self.model_string = None
        self.logger = setup_logger(name=__name__)

        self.parameter_dictionary = parameter_dictionary

        # if 'use_secondary_steering' in self.parameter_dictionary:
        #     self.use_secondary_steering = \
        #         bool(self.parameter_dictionary["use_secondary_steering"])
        # else:
        #     self.logger.info('Using default option of applying gch-based ' + \
        #                 'secondary steering (use_secondary_steering=True)')
        #     self.use_secondary_steering = True

        # if 'eps_gain' in self.parameter_dictionary:
        #     self.eps_gain = bool(self.parameter_dictionary["eps_gain"])
        # else:
        #     # SOWFA SETTING (note this will be multiplied by D in function)
        #     self.eps_gain = 0.3
        #     self.logger.info(
        #         ('Using default option eps_gain: %.1f' % self.eps_gain))

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

    def calculate_effective_yaw_angle(self, x_locations, y_locations,
                                      z_locations, turbine, coord, flow_field):
        if self.use_secondary_steering:
            # turbine parameters
            Ct = turbine.Ct
            D = turbine.rotor_diameter
            HH = turbine.hub_height
            aI = turbine.aI
            TSR = turbine.tsr

            V = flow_field.v
            W = flow_field.w

            yLocs = y_locations - coord.x2
            zLocs = z_locations - (HH)

            # Use set value
            eps = self.eps_gain * D
            # TODO Is this right below?
            Uinf = np.mean(flow_field.wind_map.input_speed)

            dist = np.sqrt(yLocs**2 + zLocs**2)
            xLocs = np.abs(x_locations - coord.x1)
            idx = np.where((dist < D / 2) & (xLocs < D / 4) \
                            & (np.abs(yLocs) > 0.1))

            Gamma = V[idx] * ((2 * np.pi) * (yLocs[idx]**2 + zLocs[idx]**2)) \
                / (yLocs[idx] * (1 - np.exp(-(yLocs[idx]**2 + zLocs[idx]**2) \
                / ((eps)**2))))
            Gamma_wake_rotation = 1.0 * 2 * np.pi * D * (aI - aI**2) \
                                  * turbine.average_velocity / TSR
            Gamma0 = np.mean(np.abs(Gamma))

            test_gamma = np.linspace(-30, 30, 61)
            minYaw = 10000
            for i in range(len(test_gamma)):
                tmp1 = 8 * Gamma0 / (np.pi * flow_field.air_density * D \
                       * turbine.average_velocity * Ct)
                tmp = np.abs((sind(test_gamma[i]) * cosd(test_gamma[i])**2) \
                      - tmp1)
                if tmp < minYaw:
                    minYaw = tmp
                    idx = i
            try:
                yaw_effective = test_gamma[idx]
            except:
                # TODO: should this really be handled or let the error throw
                # when this doesnt work?
                print('ERROR', idx)
                yaw_effective = 0.0

            return yaw_effective + turbine.yaw_angle

        else:
            return turbine.yaw_angle

    @property
    def use_secondary_steering(self):
        return self._use_secondary_steering

    @use_secondary_steering.setter
    def use_secondary_steering(self, value):
        if type(value) is not bool:
            err_msg = "Value of use_secondary_steering must be type " + \
                      "bool; {} given.".format(type(value))
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._use_secondary_steering = value

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
