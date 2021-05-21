# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from src.wake_velocity.base_velocity_deficit import VelocityDeficit
import numpy as np
from src.farm import Farm
from src.flow_field import FlowField
from src.grid import TurbineGrid

# from .base_velocity_deficit import VelocityDeficit


class JensenVelocityDeficit():
    """
    The Jensen model computes the wake velocity deficit based on the classic
    Jensen/Park model :cite:`jvm-jensen1983note`.
        
    -   **we** (*float*): The linear wake decay constant that
        defines the cone boundary for the wake as well as the
        velocity deficit. D/2 +/- we*x is the cone boundary for the
        wake.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: jvm-
    """

    default_parameters = {"we": 0.05}
    model_string = "jensen"

    def __init__(self, parameters: dict, grid: TurbineGrid, farm: Farm, flow_field: FlowField):
        self.grid = grid
        self.we = JensenVelocityDeficit.default_parameters["we"]
        self.b = flow_field.reference_turbine_diameter / 2.0

    # @staticmethod
    def function(self, i: int, farm: Farm, flow_field: FlowField, deflection) -> None:

        # grid = TurbineGrid(farm.coords, flow_field.reference_turbine_diameter, flow_field.reference_wind_height, 5)
        # flow_field.initialize_velocity_field(grid)

        # Turbine axial induction
        turbine_ai = 0.25790121826746754

        # grid.rotate_fields(flow_field.wind_directions)  # TODO: check the rotations with multiple directions or non-0/270

        # Calculate and apply wake mask
        # x = grid.x #mesh_x_rotated - x_coord_rotated

        # m = we
        # b = flow_field.reference_turbine_diameter / 2.0
        # c = (flow_field.reference_turbine_diameter / (2 * we * x + flow_field.reference_turbine_diameter)) ** 2
        
        # This is the velocity deficit seen by the i'th turbine due to wake effects from upstream turbines.
        # Indeces of velocity_deficit corresponding to unwaked turbines will have 0's
        # velocity_deficit = np.zeros(np.shape(flow_field.u_initial))

        # y = m * x + b
        boundary_line = self.we * self.grid.x[i] + self.b
    
        y_upper = boundary_line + self.grid.y[i] # + deflection_field
        y_lower = -1 * boundary_line + self.grid.y[i] # + deflection_field
        z_upper = boundary_line + flow_field.reference_wind_height
        z_lower = -1 * boundary_line + flow_field.reference_wind_height

        c = (flow_field.reference_turbine_diameter / (2 * self.we * (self.grid.x[i] - self.grid.x[i-1]) + flow_field.reference_turbine_diameter)) ** 2
        # c[mesh_x_rotated - x_coord_rotated < 0] = 0
        c[self.grid.y[i] > y_upper] = 0
        c[self.grid.y[i] < y_lower] = 0
        c[self.grid.z[i] > z_upper] = 0
        c[self.grid.z[i] < z_lower] = 0

        flow_field.u[i] = flow_field.u[i-1] * (1 - 2 * turbine_ai * c)
