"""
TurboparkGauss wake model implementation.
Refactored from standalone solver functions to BaseWakeModel-derived class.
"""
import copy

import numpy as np
from attrs import define, field, fields

from floris.core import (
    average_velocity,
    BaseModel,
    Farm,
    FlowField,
    FlowFieldPlanarGrid,
    Grid,
    PointsGrid,
    thrust_coefficient,
    TurbineGrid,
)
from floris.core.wake_velocity.gauss import gaussian_function
from floris.core.wake_velocity.turboparkgauss import TurboparkgaussVelocityDeficit
from floris.core.wake_model import BaseWakeModel

NUM_EPS = fields(BaseModel).NUM_EPS.default

@define
class TurboparkGauss(BaseWakeModel):
    """
    TurboparkGauss wake model with Gaussian wake profile.

    Model based on TurbOPark with Gaussian wake profile (Pedersen et al. 2020).
    Uses:
    - Frandsen turbulence-dependent wake expansion
    - SOSFS combination (sum of squares freestream superposition)
    - No deflection model (yaw not supported)
    - No turbulence model (ambient turbulence only)

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
            u_field (np.array): The base flow field.
            u_wake (np.array): The wake to apply to the base flow field.

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
        turbine_turbulence_intensity = np.repeat(
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
            turbulence_intensity_i = flow_field.turbulence_intensities[:, None, None, None]

            # Model calculations
            velocity_deficit = self.velocity_deficit(
                turbulence_intensity_i,
                thrust_coefficient_i,
                grid.x_sorted,
                grid.y_sorted,
                grid.z_sorted
            )

            wake_field = self.combination(
                wake_field,
                velocity_deficit * flow_field.u_initial_sorted
            )

            # Calculate wake overlap for wake-added turbulence (WAT)
            area_overlap = (
                np.sum(velocity_deficit * flow_field.u_initial_sorted > 0.05, axis=(2, 3))
                / (grid.grid_resolution * grid.grid_resolution)
            )
            area_overlap = area_overlap[:, :, None, None]

            flow_field.u_sorted = flow_field.u_initial_sorted - wake_field

        # Copy background turbulence intensity to flow field
        flow_field.turbulence_intensity_field_sorted = turbine_turbulence_intensity
        flow_field.turbulence_intensity_field_sorted_avg = np.mean(
            turbine_turbulence_intensity,
            axis=(2,3),
            keepdims=True
        )

    def point_solve(
        self,
        farm: Farm,
        flow_field: FlowField,
        grid: Grid,
    ) -> None:
        """
        Solve for visualization grid using TurboparkGauss model.
        
        Runs the same sequential deficit calculation on the full grid
        to get the visualization/output grid velocity field.
        """
        # Create velocity deficit model instance
        velocity_model = TurboparkgaussVelocityDeficit(A=self.A, include_mirror_wake=self.include_mirror_wake)
        
        # Prepare function arguments
        deficit_model_args = velocity_model.prepare_function(grid, flow_field)

        # Initialize wake state
        wake_field = np.zeros_like(flow_field.u_initial_sorted)
        v_wake = np.zeros_like(flow_field.v_initial_sorted)
        w_wake = np.zeros_like(flow_field.w_initial_sorted)

        # Set up turbulence intensity arrays
        turbine_turbulence_intensity = flow_field.turbulence_intensities[:, None, None, None]
        turbine_turbulence_intensity = np.repeat(
            turbine_turbulence_intensity, farm.n_turbines, axis=1
        )

        # Get turbine yaw angles and power setpoints from unsorted arrays to avoid sorted_indices issues
        yaw_angles_unsorted = farm.yaw_angles
        power_setpoints_unsorted = farm.power_setpoints
        awc_modes_unsorted = farm.awc_modes
        awc_amplitudes_unsorted = farm.awc_amplitudes

        # Calculate velocity deficit sequentially from upstream to downstream turbines
        for i in range(farm.n_turbines):
            
            # Get the current turbine position
            x_i = farm.coordinates[i, 0:1, None, None]
            y_i = farm.coordinates[i, 1:2, None, None]
            z_i = farm.hub_heights[i:i+1, None, None]
            hub_height_i = farm.hub_heights[i:i+1, None, None, None]

            # Get rotor diameter for this turbine
            rotor_diameter_i = farm.rotor_diameters[i:i+1, None, None, None]

            # Get yaw angle for this turbine
            yaw_angle_i = yaw_angles_unsorted[:, i:i+1, None, None]
            
            # For thrust coefficient, we need to compute on the current sorted flow field
            # But we can't use sorted properties. Instead, we recompute using unsorted arrays
            # Use defaults for grid properties that may not exist on all grid types
            avg_method = getattr(grid, 'average_method', 'max')
            cub_weights = getattr(grid, 'cubature_weights', None)
            
            ct_all = thrust_coefficient(
                turbines=farm.turbines,
                velocities=flow_field.u_sorted,
                turbulence_intensities=flow_field.turbulence_intensity_field_sorted,
                air_density=flow_field.air_density,
                yaw_angles=yaw_angles_unsorted,
                power_setpoints=power_setpoints_unsorted,
                awc_modes=awc_modes_unsorted,
                awc_amplitudes=awc_amplitudes_unsorted,
                turbine_type_map=np.broadcast_to(
                    np.array([t.turbine_type for t in farm.turbines]),
                    (flow_field.u_sorted.shape[0], farm.n_turbines)
                ),
                average_method=avg_method,
                cubature_weights=cub_weights,
                multidim_condition=flow_field.multidim_conditions,
            )
            ct_i = ct_all[:, i:i+1, None, None]

            # Get turbulence intensity for this turbine
            turbulence_intensity_i = turbine_turbulence_intensity[:, i:i+1]

            # Compute velocity deficit using TurboparkGauss velocity deficit model
            velocity_deficit = velocity_model.function(
                x_i,
                y_i,
                z_i,
                np.zeros_like(ct_i),  # axial_induction_i (not used)
                np.zeros_like(flow_field.u_initial_sorted),  # deflection_field_i (not used)
                np.zeros_like(x_i),  # yaw_angle_i (not used)
                turbulence_intensity_i,
                ct_i,
                hub_height_i,
                rotor_diameter_i,
                **deficit_model_args,
            )

            # Combine with existing wake field using SOSFS (hypot)
            wake_field = np.hypot(
                wake_field, velocity_deficit * flow_field.u_initial_sorted
            )

            # Update flow field
            flow_field.u_sorted = flow_field.u_initial_sorted - wake_field
            flow_field.v_sorted += v_wake
            flow_field.w_sorted += w_wake

        flow_field.turbulence_intensity_field_sorted = turbine_turbulence_intensity
        flow_field.turbulence_intensity_field_sorted_avg = np.mean(
            turbine_turbulence_intensity, axis=(2, 3), keepdims=True
        )


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
