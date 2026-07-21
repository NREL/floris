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
    FlowFieldPlanarGrid,
    PointsGrid,
    TurbineGrid,
)
from floris.core.wake_model import BaseWakeModel
from floris.core.wake_model.gauss import gaussian_function


NUM_EPS = fields(BaseModel).NUM_EPS.default

@define
class TurbOParkGauss(BaseWakeModel):
    """
    Model based on TurbOPark with Gaussian wake profile (Pedersen et al. 2020).

    Uses:
    - SOSFS combination (sum of squares freestream superposition)
    - No deflection model (yaw not supported)
    - No turbulence model (built into Frandsen-based wake width calculation)

    References:
        Pedersen J G, Svensen E, Poulsen L, and Nygaard N G. "Turbulence Optimized
        Park model with Gaussian wake profile." Journal of Physics: Conference
        Series. Vol. 2265. No. 022063. IOP Publishing, 2020.
        doi:10.1088/1742-6596/2265/2/022063
    """

    # TurboparkGauss-specific parameters
    A: float = field(converter=float, default=0.04)
    include_mirror_wake: bool = field(converter=bool, default=True)

    # Set during solve routines
    freestream_velocity: np.ndarray = field(init=False, default=None)


    def velocity_deficit(
        self,
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        # Initialize the velocity deficit array
        velocity_deficit = np.zeros_like(self.freestream_velocity)

        downstream_mask = (x - self.x_i >= NUM_EPS)
        x_dist = (x - self.x_i) * downstream_mask / self.rotor_diameter_i

        # Characteristic wake widths from all turbines relative to turbine i
        sigma = characteristic_wake_width(
            x_dist, turbulence_intensity_i, ct_i, self.A
        ) * self.rotor_diameter_i

        # Peak wake deficits
        C = 1 - np.sqrt(np.clip(1 - ct_i / (8 * (sigma / self.rotor_diameter_i) ** 2), 0.0, 1.0))

        r_dist = np.sqrt((y - self.y_i) ** 2 + (z - self.z_i) ** 2)

        # Compute deficits for real turbines and for mirrored (image) turbines
        delta_real  = (x_dist > 0) * gaussian_function(C, r_dist, 2, sigma)
        if self.include_mirror_wake:
            r_dist_image = np.sqrt((y - self.y_i) ** 2 + (z - 3*self.z_i) ** 2)
            delta_image = (x_dist > 0) * gaussian_function(C, r_dist_image, 2, sigma)
            delta = np.hypot(delta_real, delta_image)
        else: # No mirror wakes
            delta = delta_real

        velocity_deficit = np.nan_to_num(delta)

        return velocity_deficit

    def combination(self, wake_field: np.ndarray, velocity_field: np.ndarray):
        """
        Combines the base flow field with the velocity deficits
        using sum of squares.

        Args:
            wake_field (np.array): The velocity deficits from the wake.
            velocity_field (np.array): The base flow field.

        Returns:
            np.array: The resulting flow field after applying the wake to the
                base.
        """
        return np.hypot(wake_field, velocity_field)

    def turbine_solve(
        self,
        farm: Farm,
        flow_field: FlowField,
        grid: TurbineGrid,
    ) -> None:

        wake_field = np.zeros_like(flow_field.u_initial_sorted)

        # Expand input turbulence intensity to 4d for (n_turbines, grid, grid)
        ambient_turbulence_intensities = np.repeat(
            flow_field.turbulence_intensities[:, None, None, None],
            farm.n_turbines,
            axis=1
        )

        # Copy uniform flow field parameters
        self.freestream_velocity = flow_field.u_initial_sorted

        # Calculate the velocity deficit sequentially from upstream to downstream turbines
        for i in range(grid.n_turbines):

            # Turbine quantities
            self.set_turbine_i(grid, farm, i)
            thrust_coefficient_i = self.turbine_thrust_coefficient(grid, farm, flow_field, i)

            # Model calculations
            velocity_deficit = self.velocity_deficit(
                ambient_turbulence_intensities[:, i:i+1, :, :],
                thrust_coefficient_i,
                grid.x_sorted,
                grid.y_sorted,
                grid.z_sorted
            )

            wake_field = self.combination(
                wake_field,
                velocity_deficit * flow_field.u_initial_sorted
            )

            flow_field.u_sorted = flow_field.u_initial_sorted - wake_field

        # Copy background turbulence intensity to flow field
        flow_field.turbulence_intensity_field_sorted = ambient_turbulence_intensities
        flow_field.turbulence_intensity_field_sorted_avg = np.mean(
            ambient_turbulence_intensities,
            axis=(2,3),
            keepdims=True
        )

    def point_solve(
        self,
        farm: Farm,
        flow_field: FlowField,
        grid: FlowFieldPlanarGrid | PointsGrid,
    ) -> None:

        # Get the flow quantities and turbine performance
        (
            turbine_grid_farm,
            turbine_grid_flow_field,
            turbine_grid
        ) = self.generate_turbine_grid_objects(farm, flow_field)

        self.turbine_solve(turbine_grid_farm, turbine_grid_flow_field, turbine_grid)

        wake_field = np.zeros_like(flow_field.u_initial_sorted)

        # Initialize the turbulence intensity field over the entire flow field grid
        n_points = grid.x_sorted.shape[1]
        ambient_turbulence_intensities = flow_field.turbulence_intensities[:, None, None, None]
        ambient_turbulence_intensities = np.repeat(ambient_turbulence_intensities, n_points, axis=1)

        # Calculate the velocity deficit in the full grid sequentially from upstream to
        # downstream turbines
        for i in range(grid.n_turbines):

            # Get the current turbine quantities
            self.set_turbine_i(turbine_grid, turbine_grid_farm, i)
            thrust_coefficient_i = self.turbine_thrust_coefficient(
                turbine_grid,
                turbine_grid_farm,
                turbine_grid_flow_field,
                i
            )

            # Model calculations
            velocity_deficit = self.velocity_deficit(
                ambient_turbulence_intensities[:, i:i+1, :, :],
                thrust_coefficient_i,
                grid.x_sorted,
                grid.y_sorted,
                grid.z_sorted
            )

            wake_field = self.combination(
                wake_field,
                velocity_deficit * flow_field.u_initial_sorted
            )

            flow_field.u_sorted = flow_field.u_initial_sorted - wake_field

        flow_field.turbulence_intensity_field_sorted = ambient_turbulence_intensities


def characteristic_wake_width(x_D, ambient_TI, Cts, A):
    # Parameter values taken from S. T. Frandsen, “Risø-R-1188(EN) Turbulence
    # and turbulence generated structural loading in wind turbine clusters”
    # Risø, Roskilde, Denmark, 2007.
    c1 = 1.5
    c2 = 0.8

    alpha = ambient_TI * c1
    beta = c2 * ambient_TI / np.sqrt(Cts)

    # Term for the initial width at the turbine location (denoted epsilon in Pedersen et al.)
    # Saturate term in initial width to 3.0, as is done in Orsted Matlab code.
    initial_width = 0.25 * np.sqrt(np.minimum(0.5 * (1 + np.sqrt(1 - Cts)) / np.sqrt(1 - Cts), 3.0))

    # Term for the added width downstream of the turbine
    added_width = A * ambient_TI / beta * (
        np.sqrt((alpha + beta * x_D) ** 2 + 1)
        - np.sqrt(1 + alpha ** 2)
        - np.log(
            ((np.sqrt((alpha + beta * x_D) ** 2 + 1) + 1) * alpha)
            / ((np.sqrt(1 + alpha ** 2) + 1) * (alpha + beta * x_D))
        )
    )

    sigma_w_D = initial_width + added_width

    return sigma_w_D
