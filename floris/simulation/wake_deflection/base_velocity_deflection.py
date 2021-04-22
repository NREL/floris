# Copyright 2021 NREL

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


class VelocityDeflection(LoggerBase):
    """
    This is the super-class for all wake deflection models. It includes
    implementations of functions that subclasses should use to perform
    secondary steering. See :cite:`bvd-King2019Controls`. for more details on
    how secondary steering is calculated.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: bvd-
    """

    def __init__(self, parameter_dictionary):
        """
        Stores the parameter dictionary for the wake deflection model.

        Args:
            parameter_dictionary (dict): Dictionary containing the wake
                deflection model parameters. See individual wake deflection
                models for details of specific key-value pairs.
        """

        self.model_string = None

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
                    err_msg = (
                        "User supplied value {}, not in standard "
                        + "wake velocity model dictionary."
                    ).format(key)
                    self.logger.warning(err_msg, stack_info=True)
                    raise KeyError(err_msg)
            return_dict = user_dict
        return return_dict
