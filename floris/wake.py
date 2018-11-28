# Copyright 2017 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.


from .wake_deflection import WakeDeflection
from .wake_velocity import WakeVelocity

class Wake():
    """
    Wake is a container class for the various wake model objects. In particular,
    Wake holds references to the velocity and deflection models as well as their
    parameters.

    inputs:
        instance_dictionary: dict - the input dictionary;
            it should have the following key-value pairs:
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

    outputs:
        self: Wake - an instantiated Wake object
    """

    def __init__(self, instance_dictionary):

        self.description = instance_dictionary["description"]

        properties = instance_dictionary["properties"]
        self.velocity_model = properties["velocity_model"]
        self.deflection_model = properties["deflection_model"]
        self.parameters = properties["parameters"]

        # these attributes need special attention
        self.deflection_model = WakeDeflection(
            self.deflection_model, self.parameters)
        self.velocity_model = WakeVelocity(
            self.velocity_model, self.parameters)

    def get_deflection_function(self):
        return self.deflection_model.function

    def get_velocity_function(self):
        return self.velocity_model.function
