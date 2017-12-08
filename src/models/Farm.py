"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import matplotlib.pyplot as plt
import numpy as np
from BaseObject import BaseObject


class Farm(BaseObject):
    """
        Describe farm here
    """

    def __init__(self):
        super().__init__()
        self.turbineMap = None
        self.flowField = None

    def valid(self):
        """
            Do validity check
        """
        valid = True
        if not super().valid():
            valid = False
        return valid

    def initialize(self):
        if self.valid():
            self.flowfield.setTurbineMap(turbineMap)

    def get_flow_field(self):
        return self.flowField

    def plot_layout(self):
        plt.figure()
        x = [turbine[0] for turbine in self.turbineMap]
        y = [turbine[1] for turbine in self.turbineMap]
        plt.plot(x, y, 'o', color='g')
        plt.show()
