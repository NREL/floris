"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from .BaseObject import BaseObject
from .WakeDeflection import WakeDeflection
from .WakeVelocity import WakeVelocity

class Wake(BaseObject):

    def __init__(self, instance_dictionary):

        super().__init__()

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
