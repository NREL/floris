"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from BaseObject import BaseObject


class Controls(BaseObject):
    """
        Describe Controls here
    """

    def __init__(self,
                 turbine_angle=None,
                 yaw_angle=None,
                 tilt_angle=None,
                 blade_pitch=None):
        super().__init__()
        self.turbineAngle = turbine_angle
        self.yawAngle = yaw_angle
        self.tiltAngle = tilt_angle
        self.bladePitch = blade_pitch

    def valid(self):
        """
            Do validity check
        """
        valid = True
        if not super().valid():
            valid = False
        return valid
