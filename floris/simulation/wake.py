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
        parameters = properties["parameters"]

        self.velocity_models = {
            "jensen": wake_velocity.Jensen(parameters),
            "multizone": wake_velocity.MultiZone(parameters),
            "gauss": wake_velocity.Gauss(parameters),
            "curl": wake_velocity.Curl(parameters)
        }
        self._velocity_model = self.velocity_models[
            properties["velocity_model"]]

        self.deflection_models = {
            "jimenez": wake_deflection.Jimenez(parameters),
            "gauss": wake_deflection.Gauss(parameters),
            "curl": wake_deflection.Curl(parameters)
        }
        self._deflection_model = self.deflection_models[
            properties["deflection_model"]]

        self.combination_models = {
            "fls": wake_combination.FLS(),
            "sosfs": wake_combination.SOSFS()
        }
        self._combination_model = self.combination_models[
            properties["combination_model"]]

    # Getters & Setters
    @property
    def velocity_model(self):
        """
        Print or re-assign the velocity model. Recognized types:

         - jensen
         - multizone
         - gauss
         - curl
        """
        return self._velocity_model

    @velocity_model.setter
    def velocity_model(self, value):
        self._velocity_model = self.velocity_models[value]

    @property
    def deflection_model(self):
        """
        Print or re-assign the deflection model. Recognized types:

         - jimenez
         - gauss
         - curl
        """
        return self._deflection_model

    @deflection_model.setter
    def deflection_model(self, value):
        self._deflection_model = self.deflection_models[value]

    @property
    def combination_model(self):
        """
        Print or re-assign the combination model. Recognized types:

         - fls
         - sosfs
        """
        return self._combination_model

    @combination_model.setter
    def combination_model(self, value):
        self._combination_model = self.combination_models[value]

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
