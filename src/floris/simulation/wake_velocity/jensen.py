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

from attrs import define, field
import numpy as np

from floris.simulation import BaseModel
from floris.simulation import Farm
from floris.simulation import FlowField
from floris.simulation import Grid
from floris.simulation import Turbine


@define
class JensenVelocityDeficit(BaseModel):
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

    we: float = field(converter=float, default=0.05)
    model_string = "jensen"

    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
        turbine: Turbine
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
            reference_rotor_diameter=turbine.rotor_diameter
        )
        return kwargs

    # @profile
    def function(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        z_i: np.ndarray,
        axial_induction_i: np.ndarray,
        deflection_field_i: np.ndarray,
        yaw_angle_i: np.ndarray,
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
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

        reference_rotor_radius = reference_rotor_diameter / 2.0

        dx = x - x_i
        dy = y - y_i - deflection_field_i
        dz = z - z_i

        # y = m * x + b
        boundary_line = self.we * dx + reference_rotor_radius

        # Calculate the wake velocity deficit ratios
        # Do we need to do masking here or can it be handled in the solver?
        # TODO: why do we need to slice with i:i+1 below? This became a problem when adding the wind direction dimension. Prior to that, the dimensions worked out simply with i
        c = ( reference_rotor_radius / ( reference_rotor_radius + self.we * dx ) ) ** 2
        # c *= ~(np.array(x - x[:, :, i:i+1] <= 0.0))  # using this causes nan's in the upstream turbine because it negates the mask rather than setting it to 0. When self.we * (x - x[:, :, i:i+1]) ) == the radius, c goes to infinity and then this line flips it to Nans rather than setting to 0.
        # c *= ~(((y - y_center) ** 2 + (z - z_center) ** 2) > (boundary_line ** 2))
        # np.nan_to_num

        # C should be 0 at the current turbine and everywhere in front of it
        upstream_mask = np.array(dx <= 0.1, dtype=bool)
        # C should be 0 everywhere outside of the lateral and vertical bounds defined by the wake expansion parameter
        boundary_mask = np.array( np.sqrt(dy ** 2 + dz ** 2) > boundary_line, dtype=bool)

        mask = np.logical_or(upstream_mask, boundary_mask)
        c[mask] = 0.0

        velocity_deficit = 2 * axial_induction_i * c

        return velocity_deficit
