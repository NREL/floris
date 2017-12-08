"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from BaseObject import BaseObject


class Wake(BaseObject):

    def __init__(self):
        super().__init__()
        self.deflectionModel = None  # type: WakeDeflection
        self.velocityModel = None    # type: WakeVelocity

    def valid(self):
        """
            Implement property checking here
        """
        valid = True
        if not super().valid():
            valid = False
        return valid

    def initialize(self):
        if self.valid():
            print("cool")

    def get_deflection_function(self):
        return self.deflectionModel.function

    def get_velocity_function(self):
        return self.velocityModel.function

    def calculate(self, downstream_distance, turbine, turbine_coords):
        velocity_function = self.get_velocity_function()
        deflection_function = self.get_deflection_function()

        # the velocity deficit in general needs to know the wake deflection
        # calculate the deflection first 

        return velocity(downstream_distance, turbine_diameter, turbine_x), deflection(downstream_distance, turbine_ct, turbine_diameter)
