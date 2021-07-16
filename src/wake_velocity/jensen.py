# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from typing import Any, Dict

import attr
import numpy as np
from src.farm import Farm
from src.grid import TurbineGrid
from src.utilities import float_attrib, model_attrib
from src.base_class import BaseClass
from src.flow_field import FlowField


@attr.s(auto_attribs=True)
class JensenVelocityDeficit(BaseClass):
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

    we: float = float_attrib(default=-0.05)
    model_string: str = model_attrib(default="jensen")

    def prepare_function(
        deflection_field: np.array, grid: TurbineGrid, farm: Farm, flow_field: FlowField
    ) -> Dict[str, Any]:
        kwargs = dict(
            deflection_field=deflection_field,
            x=grid.x,
            y=grid.y,
            z=grid.z,
            u=flow_field.u,
            reference_wind_height=flow_field.reference_wind_height,
            reference_turbine_diameter=flow_field.reference_turbine_diameter,
        )
        return kwargs

    def function(
        self,
        i: int,  # need to flesh this and deflection field out more
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        reference_wind_height: float,
        reference_turbine_diameter: float,
    ) -> None:

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
        boundary_line = self.we * x[i] + self.b

        y_upper = boundary_line + y[i]  # + deflection_field
        y_lower = -1 * boundary_line + y[i]  # + deflection_field
        z_upper = boundary_line + reference_wind_height
        z_lower = -1 * boundary_line + reference_wind_height

        c = (
            reference_turbine_diameter
            / (2 * self.we * (x[i] - x[i - 1]) + reference_turbine_diameter)
        ) ** 2
        # c[mesh_x_rotated - x_coord_rotated < 0] = 0
        c[y[i] > y_upper] = 0
        c[y[i] < y_lower] = 0
        c[z[i] > z_upper] = 0
        c[z[i] < z_lower] = 0

        u[i] = u[i - 1] * (1 - 2 * turbine_ai * c)


# J = JensenVelocityDeficit()
# function_kwargs = J.prepare_function(grid, farm, field)
# J.function(i, df, **function_kwargs)
