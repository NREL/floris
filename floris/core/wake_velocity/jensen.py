
from typing import Any, Dict

import numexpr as ne
import numpy as np
from attrs import (
    define,
    field,
    fields,
)

from floris.core import (
    BaseModel,
    Farm,
    FlowField,
    Grid,
    Turbine,
)


NUM_EPS = fields(BaseModel).NUM_EPS.default

@define
class JensenVelocityDeficit(BaseModel):
    """
    The Jensen model computes the wake velocity deficit based on the classic
    Jensen/Park model :cite:`jvm-jensen1983note`.

    -   **we** (*float*): The linear wake decay constant that
        defines the cone boundary for the wake as well as the
        velocity deficit. D/2 +/- we*x is the cone boundary for the
        wake. Used when we_kb=0 (constant expansion mode).

    -   **we_ka** (*float*): Baseline wake expansion coefficient (k_a).
        When we_kb is non-zero, effective we = we_ka + we_kb * TI.
        Default is 0.0.

    -   **we_kb** (*float*): TI-dependent wake expansion coefficient (k_b).
        When non-zero, enables TI-dependent expansion. Default is 0.0.

    References:
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: jvm-
    """

    we: float = field(converter=float, default=0.05)
    we_ka: float = field(converter=float, default=0.0)
    we_kb: float = field(converter=float, default=0.0)
    use_ambient_ti: bool = field(default=False)

    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> Dict[str, Any]:
        """
        This function prepares the inputs from the various FLORIS data structures
        for use in the Jensen model. This should only be used to 'initialize'
        the inputs. For any data that should be updated successively,
        do not use this function and instead pass that data directly to
        the model function.
        """
        kwargs = {
            "x": grid.x_sorted,
            "y": grid.y_sorted,
            "z": grid.z_sorted,
        }
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
        hub_height_i,
        rotor_diameter_i,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> None:

        # u is 4-dimensional (n wind speeds, n turbines, grid res 1, grid res 2)
        # velocities is 3-dimensional (n turbines, grid res 1, grid res 2)

        # TODO: check the rotations with multiple directions or non-0/270
        # grid.rotate_fields(flow_field.wind_directions)

        # Calculate and apply wake mask
        # x = grid.x_sorted # mesh_x_rotated - x_coord_rotated

        # This is the velocity deficit seen by the i'th turbine due to wake effects
        # from upstream turbines.
        # Indeces of velocity_deficit corresponding to unwaked turbines will have 0's
        # velocity_deficit = np.zeros(np.shape(flow_field.u_initial))

        rotor_radius = rotor_diameter_i / 2.0

        # Numexpr - do not change below without corresponding changes above.
        dx = ne.evaluate("x - x_i")
        dy = ne.evaluate("y - y_i - deflection_field_i")
        dz = ne.evaluate("z - z_i")

        # Compute effective wake expansion: we_eff = we_ka + we_kb * TI
        # If we_kb=0, fall back to constant we

        if (self.we_kb != 0.0) or (self.we_ka != 0.0):
            we = self.we_ka + self.we_kb * turbulence_intensity_i
            # print(f"Using TI-dependent wake expansion: we = {self.we_ka} + {self.we_kb} * TI = {we}")
        else:
            we = self.we
            # print(f"Using constant wake expansion: we = {we}")
            
        # Construct a boolean mask to include all points downstream of the turbine
        downstream_mask = ne.evaluate("dx > 0 + NUM_EPS")
        # Construct a boolean mask to include all points within the wake boundary
        # as defined by the Jensen model. This is a linear wake expansion that makes
        # a shape like a cone and starts at the turbine disc.
        # The left side of the inequality below evaluates the distance from the wake centerline
        # for all points including positive and negative values. The inequality compares distance
        # from the centerline and it must be below the line defined by the wake
        # expansion parameter, "we".
        boundary_mask = ne.evaluate("sqrt(dy ** 2 + dz ** 2) < we * dx + rotor_radius")

        # Calculate C for points within the mask and fill points outside with 0
        c = np.where(
            np.logical_and(downstream_mask, boundary_mask),
            ne.evaluate("(rotor_radius / (rotor_radius + we * dx + NUM_EPS)) ** 2"),  # This is "C"
            0.0,
        )

        velocity_deficit = ne.evaluate("2 * axial_induction_i * c")

        return velocity_deficit
