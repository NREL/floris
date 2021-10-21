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

from src.turbine import Turbine
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

    we: float = float_attrib(default=0.05)
    model_string: str = model_attrib(default="jensen")

    def prepare_function(
        self,
        grid: TurbineGrid,
        reference_turbine: Turbine,
        flow_field: FlowField
    ) -> Dict[str, Any]:
        """
        This function prepares the inputs from the various FLORIS data structures
        for use in the Jensen model. This should only be used to 'initialize'
        the inputs. For any data that should be updated successively,
        do not use this function and instead pass that data directly to
        the model function.
        """
        kwargs = dict(
            x=grid.x,
            y=grid.y,
            z=grid.z,
            reference_wind_height=flow_field.reference_wind_height,
            reference_turbine=reference_turbine,
        )
        return kwargs

    def function(
        self,
        i: int,
        deflection_field: np.ndarray,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        reference_wind_height: float,
        reference_turbine: Turbine,
    ) -> None:

        # u is 4-dimensional (n wind speeds, n turbines, grid res 1, grid res 2)
        # velocities is 3-dimensional (n turbines, grid res 1, grid res 2)

        # grid.rotate_fields(flow_field.wind_directions)  # TODO: check the rotations with multiple directions or non-0/270

        # Calculate and apply wake mask
        # x = grid.x # mesh_x_rotated - x_coord_rotated
        
        # This is the velocity deficit seen by the i'th turbine due to wake effects from upstream turbines.
        # Indeces of velocity_deficit corresponding to unwaked turbines will have 0's
        # velocity_deficit = np.zeros(np.shape(flow_field.u_initial))

        m = self.we
        # x = x[i] - x[i - 1] #mesh_x_rotated - x_coord_rotated
        b = reference_turbine.rotor_diameter / 2.0

        boundary_line = m * x + b

        y_center = np.zeros_like(boundary_line) + y + deflection_field
        z_center = np.zeros_like(boundary_line) + reference_wind_height

        # Calculate the wake velocity deficit ratios
        c = (
            (reference_turbine.rotor_diameter / (2 * self.we * (x - x[i]) + reference_turbine.rotor_diameter)) ** 2
            * ~(np.array(x - x[i] <= 0.0))  # using this causes nan's in the upstream turbine
            * ~(((y - y_center) ** 2 + (z - z_center) ** 2) > (boundary_line ** 2))
        )

        # c[x - x[i] <= 0] = 0
        # mask = (((y - y_center) ** 2 + (z - z_center) ** 2) ** 2) > (boundary_line ** 2)
        # c[mask] = 0

        return c
        # u[i] = u[i - 1] * (1 - 2 * turbine_ai * c)

        # This combination model is essentially the freestream linear superposition of v2
        # This is used in the original paper.
        # flow_field.u[i] = flow_field.u[i-1] * (1 - 2 * turbine_ai * c)
