# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from . import wake_deflection
from . import wake_velocity
from . import wake_combination


class Wake():
    """
    Wake is a container class for the various wake model objects. In
    particular, Wake holds references to the velocity and deflection
    models as well as their parameters.
    """

    def __init__(self, instance_dictionary):
        """
        Init method for Wake objects.

        Args:
            instance_dictionary (dict): the input dictionary with the
            following key-value pairs:
                {
                    "description": str,
                    "properties": dict({
                        velocity_model: WakeVelocity
                        deflection_model: WakeDeflection
                        parameters: dict({
                            see WakeVelocity, WakeDeflection
                        })
                    }),
                }
        """

        self.description = instance_dictionary["description"]
        properties = instance_dictionary["properties"]
        # TODO: Support dynamically setting the parameters; currently, once theyre
        #   set after instantiating FLORIS, it would be difficult to reset and to
        #   know that they should be reset
        self.parameters = properties["parameters"]

        self._velocity_models = {
            "jensen": wake_velocity.Jensen,
            "multizone": wake_velocity.MultiZone,
            "gauss": wake_velocity.Gauss,
            "curl": wake_velocity.Curl,
            "gauss_curl_hybrid": wake_velocity.GaussCurlHybrid
        }
        self.velocity_model = properties["velocity_model"]

        self._deflection_models = {
            "jimenez": wake_deflection.Jimenez,
            "gauss": wake_deflection.Gauss,
            "curl": wake_deflection.Curl,
            "gauss_curl_hybrid": wake_deflection.GaussCurlHybrid
        }
        self.deflection_model = properties["deflection_model"]

        self._combination_models = {
            "fls": wake_combination.FLS,
            "sosfs": wake_combination.SOSFS
        }
        self.combination_model = properties["combination_model"]

    # Getters & Setters
    @property
    def velocity_model(self):
        """
        Print or re-assign the velocity model. Recognized types:

         - jensen
         - multizone
         - gauss
         - curl
         - gauss_curl_hybrid

        When assigning, the input can be a string or an instance of the model.
        """
        return self._velocity_model

    @velocity_model.setter
    def velocity_model(self, value):
        if type(value) is str:
            self._velocity_model = self._velocity_models[value](self.parameters)
        elif isinstance(value, wake_velocity.WakeVelocity):
            self._velocity_model = value
        else:
            raise ValueError("Invalid value given for WakeVelocity: {}".format(value))

    @property
    def deflection_model(self):
        """
        Print or re-assign the deflection model. Recognized types:

         - jimenez
         - gauss
         - curl
         - gauss_curl_hybrid

        When assigning, the input can be a string or an instance of the model.
        """
        return self._deflection_model

    @deflection_model.setter
    def deflection_model(self, value):
        if type(value) is str:
            self._deflection_model = self._deflection_models[value](self.parameters)
        elif isinstance(value, wake_deflection.WakeDeflection):
            self._deflection_model = value
        else:
            raise ValueError("Invalid value given for WakeDeflection: {}".format(value))

    @property
    def combination_model(self):
        """
        Print or re-assign the combination model. Recognized types:

         - fls
         - sosfs

        When assigning, the input can be a string or an instance of the model.
        """
        return self._combination_model

    @combination_model.setter
    def combination_model(self, value):
        if type(value) is str:
            self._combination_model = self._combination_models[value]()
        elif isinstance(value, wake_combination.WakeCombination):
            self._combination_model = value
        else:
            raise ValueError("Invalid value given for WakeCombination: {}".format(value))

    @property
    def deflection_function(self):
        """
        Return the underlying function of the deflection model.
        """
        return self._deflection_model.function

    @property
    def velocity_function(self):
        """
        Return the underlying function of the velocity model.
        """
        return self._velocity_model.function

    @property
    def combination_function(self):
        """
        Return the underlying function of the combination model.
        """
        return self._combination_model.function
