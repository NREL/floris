"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import os
import sys
from BaseObject import BaseObject
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.io.VisualizationManager import VisualizationManager


class FlowField(BaseObject):
    """
        Describe FF here
    """

    def __init__(self,
                 wake_combination=None,
                 wind_speed=None,
                 shear=None,
                 turbine_map=None,
                 characteristic_height=None,
                 wake=None):
        super().__init__()
        self.vizManager = VisualizationManager()
        self.wakeCombination = wake_combination
        self.windSpeed = wind_speed
        self.shear = shear
        self.turbineMap = turbine_map

        # {
        #   (x,y): {Turbine, TurbineSolution(), Wake()},
        #   (x,y): {Turbine, TurbineSolution(), Wake()},
        #   ...
        # }

        # FlowfieldPropertiesAtTurbine: {
        #     (0, 0): {
        #         Turbine,
        #         ti,
        #         coordinates,
        #         velocity,
        #         get_ct(self.velocity): return turbine.Ct,
        #         get_cp(self.velocity): return turbine.Cp,
        #         get_power,
        #         wake_function
        #     },
        #     (0,10): Turbine,
        #     (0,20): Turbine,
        # }

        self.characteristicHeight = characteristic_height
        self.wake = wake

        if self._valid():
            self._initialize_turbines()
            self.x, self.y, self.z = self._discretize_domain()
            self.u_field = self._constant_flowfield(self.windSpeed)

    def _valid(self):
        """
            Do validity check
        """
        valid = True
        if not super().valid():
            return False
        if self.characteristicHeight <= 0:
            valid = False
        return valid

    def _initialize_turbines(self):
        # TODO: this should only be applied to any turbine seeing freestream -> why? if so, then all turbines cannot be initialized
        # initialize the turbine disk velocities used in the 3D model based on shear using the power log law.
        for _, turbine in self.turbineMap.items():
            grid = turbine.get_grid()
            # use the z coordinate of the turbine grid points for initialization
            velocities = [self.windSpeed * ((turbine.hubHeight+g[1]) / 
                self.characteristicHeight)**self.shear for g in grid]
            turbine.initialize(velocities)

    def _discretize_domain(self):
        coords = [coord for coord, _ in self.turbineMap.items()]
        x = [coord.x for coord in coords]
        y = [coord.y for coord in coords]
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)

        turbines = [turbine for _, turbine in self.turbineMap.items()]
        maxDiameter = max([turbine.rotorDiameter for turbine in turbines])
        hubHeight = turbines[0].hubHeight

        x = np.linspace(xmin - 2 * maxDiameter, xmax + 10 * maxDiameter, 200)
        y = np.linspace(ymin - 2 * maxDiameter, ymax + 2 * maxDiameter, 200)
        z = np.linspace(0, 2 * hubHeight, 50)
        return np.meshgrid(x, y, z, indexing='xy')

    def _constant_flowfield(self, constant_value):
        xmax, ymax, zmax = self.x.shape[0], self.y.shape[1], self.z.shape[2]
        return np.full((xmax, ymax, zmax), constant_value)

    def _compute_turbine_velocity_deficit(self, x, y, z, turbine, coord):
        """
            computes the discrete velocity field x, y, z for turbine using velocity_function
        """
        velocity_function = self.wake.get_velocity_function()
        return velocity_function(x, y, z, turbine, coord)

    def _compute_turbine_wake_deflection(self, x, deflection_function, turbine):
        deflection_function = self.wake.get_deflection_function()
        return None

    def calculate_wake(self):
        # TODO: rotate layout here
        # TODO: sort in ascending order of x coord

        # calculate the velocities on the mesh
        u_wake = np.zeros(self.u_field.shape)
        for coord, turbine in self.turbineMap.items():
            # get the velocity deficit
            u_wake += self.windSpeed * self._compute_turbine_velocity_deficit(
                self.x, self.y, self.z, turbine, coord)

            # deflect the velocity deficit
            # u_wake = self.compute_turbine_wake_deflection(
                # x, deflection_function, turbine)

        # combine this turbine's wake into the full flow field
        self.u_field = self.wakeCombination.combine(None, None, self.u_field, u_wake)

    # Visualization
    def plot_flow_field_plane(self):
        self.vizManager.plot_constant_z(
            self.x[:, :, 24], self.y[:, :, 24], self.u_field[:, :, 24])
        for coord, turbine in self.turbineMap.items():
            self.vizManager.add_turbine_marker(turbine.rotorRadius, coord)
        self.vizManager.show_plot()

    # FUTURE
    # TODO def get_properties_at_turbine(tuple_of_coords):
    #     #probe the FlowField
    #     FlowfieldPropertiesAtTurbine[tuple_of_coords].wake_function()

    # TODO def update_flowfield():