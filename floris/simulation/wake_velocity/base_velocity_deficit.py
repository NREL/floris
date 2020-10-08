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

from ...utilities import cosd, sind, tand
from ...logging_manager import LoggerBase


class VelocityDeficit(LoggerBase):
    """
    Base VelocityDeficit object class. This class contains a method for getting
    the relevant model parameters from the input dictionary, or for supplying
    default values if none are supplied.
    """

    def __init__(self, parameter_dictionary):
        """
        Stores the parameter dictionary for the wake velocity model.

        Args:
            parameter_dictionary (dict): Dictionary containing the wake
                velocity model parameters. See individual wake velocity
                models for details of specific key-value pairs.
        """
        self.requires_resolution = False
        self.model_string = None
        self.model_grid_resolution = None
        self.parameter_dictionary = parameter_dictionary

        # TODO: verify we can remove the commented code here
        # if 'calculate_VW_velocities' in self.parameter_dictionary:
        #     self.calculate_VW_velocities = \
        #         bool(self.parameter_dictionary["calculate_VW_velocities"])
        # else:
        #     self.logger.info('Using default option of calculating V and W ' + \
        #         'velocity components (calculate_VW_velocities=False)')
        #     self.calculate_VW_velocities = True

        # if 'use_yaw_added_recovery' in self.parameter_dictionary:
        #     # if set to True, self.calculate_VW_velocities also is set to True
        #     self.use_yaw_added_recovery = \
        #         bool(self.parameter_dictionary["use_yaw_added_recovery"])
        # else:
        #     self.logger.info('Using default option of applying added ' + \
        #                 'yaw-added recovery (use_yaw_added_recovery=True)')
        #     self.use_yaw_added_recovery = True

        # if 'yaw_recovery_alpha' in self.parameter_dictionary:
        #     self.yaw_recovery_alpha = \
        #         bool(self.parameter_dictionary["yaw_recovery_alpha"])
        # else:
        #     self.yaw_recovery_alpha = 0.03
        #     self.logger.info('Using default option yaw_recovery_alpha: %.2f' \
        #                 % self.yaw_recovery_alpha)

        # if 'eps_gain' in self.parameter_dictionary:
        #     self.eps_gain = bool(self.parameter_dictionary["eps_gain"])
        # else:
        #     self.eps_gain = 0.3  # SOWFA SETTING (note this will be multiplied
        #     # by D in function)
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
                    err_msg = (
                        "User supplied value {}, not in standard "
                        + "wake velocity model dictionary."
                    ).format(key)
                    self.logger.warning(err_msg, stack_info=True)
                    raise KeyError(err_msg)
            return_dict = user_dict
        return return_dict
