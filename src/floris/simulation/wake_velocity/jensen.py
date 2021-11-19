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

from floris.simulation import TurbineGrid
from floris.simulation import Turbine
from floris.utilities import float_attrib, model_attrib
from floris.simulation import BaseClass
from floris.simulation import FlowField


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
        reference_rotor_diameter: float,
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
            reference_rotor_diameter=reference_rotor_diameter,
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
        reference_rotor_diameter: float,
    ) -> None:

        # u is 4-dimensional (n wind speeds, n turbines, grid res 1, grid res 2)
        # velocities is 3-dimensional (n turbines, grid res 1, grid res 2)

        # grid.rotate_fields(flow_field.wind_directions)  # TODO: check the rotations with multiple directions or non-0/270

        # Calculate and apply wake mask
        # x = grid.x # mesh_x_rotated - x_coord_rotated

        # This is the velocity deficit seen by the i'th turbine due to wake effects from upstream turbines.
        # Indeces of velocity_deficit corresponding to unwaked turbines will have 0's
        # velocity_deficit = np.zeros(np.shape(flow_field.u_initial))

        reference_rotor_radius = reference_rotor_diameter[:, :, :, None, None] / 2.0

        # y = m * x + b
        boundary_line = self.we * x + reference_rotor_radius

        y_i = np.mean(y[:, :, i:i+1], axis=(3,4))
        y_i = y_i[:, :, :, None, None] + deflection_field
        z_i = np.mean(z[:, :, i:i+1], axis=(3,4))
        z_i = z_i[:, :, :, None, None]

        # Calculate the wake velocity deficit ratios
        # Do we need to do masking here or can it be handled in the solver?
        # TODO: why do we need to slice with i:i+1 below? This became a problem when adding the wind direction dimension. Prior to that, the dimensions worked out simply with i
        dx = x - x[:, :, i:i+1]
        c = ( reference_rotor_radius / ( reference_rotor_radius + self.we * dx ) ) ** 2
        # c *= ~(np.array(x - x[:, :, i:i+1] <= 0.0))  # using this causes nan's in the upstream turbine because it negates the mask rather than setting it to 0. When self.we * (x - x[:, :, i:i+1]) ) == the radius, c goes to infinity and then this line flips it to Nans rather than setting to 0.
        # c *= ~(((y - y_center) ** 2 + (z - z_center) ** 2) > (boundary_line ** 2))
        # np.nan_to_num
        # C should be 0 at the current turbine and everywhere in front of it
        c[x - x[:, :, i:i+1] <= 0.0] = 0.0
        mask = ((y - y_i) ** 2 + (z - z_i) ** 2) > (boundary_line ** 2)
        c[mask] = 0.0

        return c
        # u[i] = u[i - 1] * (1 - 2 * turbine_ai * c)

        # This combination model is essentially the freestream linear superposition of v2
        # This is used in the original paper.
        # flow_field.u[i] = flow_field.u[i-1] * (1 - 2 * turbine_ai * c)
