# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

class WakeTurbulence():
    """
<<<<<<< HEAD
    WakeTurbulence is the general class for instances of specific wake-added 
    turbulence models. 
=======
    This is the super-class for all wake turbulence models. It includes
    implementations of functions that subclasses should use to retrieve
    model-specific parameters from the input dictionary.
>>>>>>> docstrings
    """
    def __init__(self, parameter_dictionary):
        """
<<<<<<< HEAD
        Initialize the general wake turbulence class with the name of a 
        specific turbulence model and the associated parameter values. If no 
        parameter values are provided, default values are used.

        Args:
            parameter_dictionary (dict): Dictionary consisting of the following key-value pairs:

                - model_string: name of turbulence model
                - parameters: see turbulence model classes
        """
        
=======
        Stores the parameter dictionary for the wake deflection model.

        Args:
            parameter_dictionary (dict): Contains the wake turbulence
                model parameters. See individual wake turbulence
                models for details of specific key-value pairs.
        """
>>>>>>> docstrings
        self.parameter_dictionary = parameter_dictionary

        self.requires_resolution = False
        self.model_string = None
        self.model_grid_resolution = None

    def __str__(self):
        return self.model_string

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