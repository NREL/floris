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
    FlowFieldPlanarGrid,
    PointsGrid,
    TurbineGrid,
)
from floris.core.rotor_velocity import (
    average_velocity,
    calculate_tilt_for_rotor_effective_velocities,
)
from floris.core.wake_model import BaseWakeModel
from floris.core.wake_model.gauss import gaussian_function
from floris.type_dec import floris_float_type
from floris.utilities import cosd


NUM_EPS = fields(BaseModel).NUM_EPS.default

@define
class EmpiricalGauss(BaseWakeModel):

    # Deficit model parameters
    wake_expansion_rates: list = field(factory=lambda: [0.023, 0.008])
    breakpoints_D: list = field(factory=lambda: [10])
    sigma_0_D: float = field(default=0.28)
    smoothing_length_D: float = field(default=2.0)
    mixing_gain_velocity: float = field(default=2.0)
    awc_mode: str = field(default="baseline")
    awc_wake_exp: float = field(default=1.2)
    awc_wake_denominator: float = field(default=400)
    include_mirror_wake: bool = field(default=True)

    # Deflection model parameters
    horizontal_deflection_gain_D: float = field(default=3.0)
    vertical_deflection_gain_D: float = field(default=-1)
    deflection_rate: float = field(default=22)
    mixing_gain_deflection: float = field(default=0.0)
    yaw_added_mixing_gain: float = field(default=0.0)

    # Mixing model parameters
    atmospheric_ti_gain: float = field(converter=float, default=0.0)
    enable_yaw_added_recovery: bool = field(default=True)
    enable_active_wake_mixing: bool = field(default=True)

    tilt_angle_i: np.ndarray = field(init=False, default=None)

    ambient_turbulence_intensities: np.ndarray = field(init=False, default=None)
    wind_veer: float = field(init=False, default=None)
    freestream_velocity: np.ndarray = field(init=False, default=None)
    mixing_factor: np.ndarray = field(init=False, default=None)

    def velocity_deficit(
        self,
        deflection_field_y_i: np.ndarray,
        deflection_field_z_i: np.ndarray,
        mixing_i: np.ndarray,
        ct_i: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:

        # Only symmetric terms using yaw, but keep for consistency
        yaw_angle = -1 * self.yaw_angle_i

        # Initial wake widths
        sigma_y0 = self.sigma_0_D * self.rotor_diameter_i * cosd(yaw_angle)
        sigma_z0 = self.sigma_0_D * self.rotor_diameter_i * cosd(self.tilt_angle_i)

        # No specific near, far wakes in this model
        downstream_mask = (x > self.x_i + 0.1)
        upstream_mask = (x < self.x_i - 0.1)

        # Wake expansion in the lateral (y) and the vertical (z)
        # TODO: could compute shared components in sigma_z, sigma_y
        # with one function call.
        sigma_y = empirical_gauss_model_wake_width(
            x - self.x_i,
            self.wake_expansion_rates,
            [b * self.rotor_diameter_i for b in self.breakpoints_D], # .flatten()[0]
            sigma_y0,
            self.smoothing_length_D * self.rotor_diameter_i,
            self.mixing_gain_velocity * mixing_i,
        )
        sigma_y[upstream_mask] = \
            np.tile(sigma_y0, np.shape(sigma_y)[1:])[upstream_mask]

        sigma_z = empirical_gauss_model_wake_width(
            x - self.x_i,
            self.wake_expansion_rates,
            [b * self.rotor_diameter_i for b in self.breakpoints_D], # .flatten()[0]
            sigma_z0,
            self.smoothing_length_D * self.rotor_diameter_i,
            self.mixing_gain_velocity * mixing_i,
        )
        sigma_z[upstream_mask] = \
            np.tile(sigma_z0, np.shape(sigma_z)[1:])[upstream_mask]

        # 'Standard' wake component
        r, C = rCalt(
            self.wind_veer,
            sigma_y,
            sigma_z,
            y,
            self.y_i,
            deflection_field_y_i,
            deflection_field_z_i,
            z,
            self.hub_height_i,
            ct_i,
            yaw_angle,
            self.tilt_angle_i,
            self.rotor_diameter_i,
            sigma_y0,
            sigma_z0
        )
        # Normalize to match end of actuator disk model tube
        C = C / (8 * self.sigma_0_D**2 )

        wake_deficit = gaussian_function(C, r, 1, np.sqrt(0.5))

        if self.include_mirror_wake:
            # TODO: speed up this option by calculating various elements in
            #       rCalt only once.
            # Mirror component
            r_mirr, C_mirr = rCalt(
                self.wind_veer, # TODO: Is veer OK with mirror wakes?
                sigma_y,
                sigma_z,
                y,
                self.y_i,
                deflection_field_y_i,
                deflection_field_z_i,
                z,
                -self.hub_height_i, # Turbine at negative hub height location
                ct_i,
                yaw_angle,
                self.tilt_angle_i,
                self.rotor_diameter_i,
                sigma_y0,
                sigma_z0
            )
            # Normalize to match end of actuator disk model tube
            C_mirr = C_mirr / (8 * self.sigma_0_D**2)

            # ASSUME sum-of-squares superposition for the real and mirror wakes
            wake_deficit = np.sqrt(
                wake_deficit**2 +
                gaussian_function(C_mirr, r_mirr, 1, np.sqrt(0.5))**2
            )

        velocity_deficit = wake_deficit * downstream_mask

        return velocity_deficit

    def deflection(
        self,
        mixing_i: np.ndarray,
        ct_i: np.ndarray,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        deflection_gain_y = self.horizontal_deflection_gain_D * self.rotor_diameter_i
        if self.vertical_deflection_gain_D == -1:
            deflection_gain_z = deflection_gain_y
        else:
            deflection_gain_z = self.vertical_deflection_gain_D * self.rotor_diameter_i

        # Convert to radians, CW yaw for consistency with other models
        yaw_r = np.pi/180 * -self.yaw_angle_i
        tilt_r = np.pi/180 * self.tilt_angle_i

        A_y = (deflection_gain_y * ct_i * yaw_r) / (1 + self.mixing_gain_deflection * mixing_i)
        A_z = (deflection_gain_z * ct_i * tilt_r) / (1 + self.mixing_gain_deflection * mixing_i)

        # Apply downstream mask in the process
        x_normalized = (x - self.x_i) * (x > self.x_i + 0.1) / self.rotor_diameter_i

        log_term = np.log(
            (x_normalized - self.deflection_rate) / (x_normalized + self.deflection_rate)
            + 2
        )

        deflection_y = A_y * log_term
        deflection_z = A_z * log_term

        return deflection_y, deflection_z

    def mixing(
        self,
        axial_induction_i: np.ndarray,
        downstream_distance_D_i: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the contribution of turbine i to all other turbines'
        mixing terms.

        Args:
            axial_induction_i (np.array): Axial induction factor of
                the ith turbine (-).
            downstream_distance_D_i (np.array): The distance downstream
                from turbine i to all other turbines (specified in terms
                of multiples of turbine i's rotor diameter) (D).

        Returns:
            np.array: Components of the wake-induced mixing term due to
                the ith turbine.
        """

        wake_induced_mixing = axial_induction_i[:,:,0,0] / downstream_distance_D_i**2

        return wake_induced_mixing

    def turbine_solve(
        self,
        farm: Farm,
        flow_field: FlowField,
        grid: TurbineGrid,
    ) -> None:

        wake_field = np.zeros_like(flow_field.u_initial_sorted)

        # Initialize mixing factor information
        x_locs = np.mean(grid.x_sorted, axis=(2, 3))[:,:,None]
        downstream_distance_D = x_locs - np.transpose(x_locs, axes=(0,2,1))
        downstream_distance_D = downstream_distance_D / \
            np.repeat(farm.rotor_diameters_sorted[:,:,None], grid.n_turbines, axis=-1)
        downstream_distance_D = np.maximum(downstream_distance_D, 0.1) # For ease
        # Initialize the mixing factor model using TI if specified
        initial_mixing_factor = self.atmospheric_ti_gain * np.eye(grid.n_turbines)
        mixing_factor = np.repeat(
            initial_mixing_factor[None, :, :],
            flow_field.n_findex,
            axis=0
        )
        mixing_factor = mixing_factor * flow_field.turbulence_intensities[:, None, None]

        # Ambient turbulent intensity should be a copy of n_findex-long turbulence_intensity
        # with dimensions expanded for (n_turbines, grid, grid)
        self.ambient_turbulence_intensities = flow_field.turbulence_intensities[:, None, None, None]

        # Copy uniform flow field parameters
        self.freestream_velocity = flow_field.u_initial_sorted
        self.wind_veer = flow_field.wind_veer


        # Calculate the velocity deficit sequentially from upstream to downstream turbines
        for i in range(grid.n_turbines):

            # Turbine quantities
            self.set_turbine_i(grid, farm, i)
            thrust_coefficient_i = self.evaluate_turbine_thrust_coefficient(
                grid, farm, flow_field, i
            )
            axial_induction_i = self.evaluate_turbine_axial_induction(grid, farm, flow_field, i)

            # Compute the tilt angle of the ith turbine
            average_velocities = average_velocity(
                flow_field.u_sorted,
                method=grid.average_method,
                cubature_weights=grid.cubature_weights
            )
            self.tilt_angle_i = calculate_tilt_for_rotor_effective_velocities(
                farm, average_velocities
            )[:, i:i+1, None, None]

            if self.enable_yaw_added_recovery:
                # Influence of yawing on turbine's own wake
                mixing_factor[:, i:i+1, i] += \
                    yaw_added_wake_mixing(
                        axial_induction_i, self.yaw_angle_i, 1, self.yaw_added_mixing_gain
                    )
            if self.enable_active_wake_mixing:
                # Influence of awc on turbine's own wake
                mixing_factor[:, i:i+1, i] += \
                    awc_added_wake_mixing(
                        farm.awc_modes_sorted[:, i:i+1, None, None],
                        farm.awc_amplitudes_sorted[:, i:i+1, None, None],
                        farm.awc_frequencies_sorted[:, i:i+1, None, None],
                        self.awc_wake_exp,
                        self.awc_wake_denominator
                    )

            # Extract total wake induced mixing for turbine i
            mixing_i = np.linalg.norm(
                mixing_factor[:, i:i+1, :, None],
                ord=2, axis=2, keepdims=True
            )

            # Primary model calculations
            deflection_field_y, deflection_field_z = self.deflection(
                mixing_i,
                thrust_coefficient_i,
                grid.x_sorted,
            )

            velocity_deficit = self.velocity_deficit(
                deflection_field_y,
                deflection_field_z,
                mixing_i,
                thrust_coefficient_i,
                grid.x_sorted,
                grid.y_sorted,
                grid.z_sorted
            )

            wake_field = self.combination_function(
                wake_field,
                velocity_deficit * flow_field.u_initial_sorted
            )

            # Calculate wake overlap for wake-added turbulence (WAT)
            area_overlap = np.sum(
                velocity_deficit * flow_field.u_initial_sorted > 0.05,
                axis=(2, 3)
            ) / (grid.grid_resolution * grid.grid_resolution)

            # Compute wake induced mixing factor
            mixing_factor[:,:,i] += area_overlap * self.mixing(
                axial_induction_i, downstream_distance_D[:,:,i]
            )

            if self.enable_yaw_added_recovery:
                mixing_factor[:,:,i] += \
                    area_overlap * yaw_added_wake_mixing(
                    axial_induction_i,
                    self.yaw_angle_i,
                    downstream_distance_D[:,:,i],
                    self.yaw_added_mixing_gain
                )

            # Remove wakes from flow field
            flow_field.u_sorted = flow_field.u_initial_sorted - wake_field

        # Store for use in point_solve
        self.mixing_factor = mixing_factor

        # Compute turbine powers based on final flow field
        self.evaluate_turbine_power(grid, farm, flow_field)

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
        turbulence_intensity_field = ambient_turbulence_intensities.copy()

        # Extract freestream velocity for deficit, deflection calculations
        self.freestream_velocity = flow_field.u_initial_sorted

        # Calculate the velocity deficit in the full grid sequentially from upstream to
        # downstream turbines
        for i in range(grid.n_turbines):

            # Get the current turbine quantities
            self.set_turbine_i(turbine_grid, turbine_grid_farm, i)
            thrust_coefficient_i = self.evaluate_turbine_thrust_coefficient(
                turbine_grid,
                turbine_grid_farm,
                turbine_grid_flow_field,
                i
            )

            # Get mixing_i based on turbine_solve results
            mixing_i = self.mixing_factor[:, i:i+1, :, None].sum(axis=2, keepdims=1)

            average_velocities = average_velocity(
                turbine_grid_flow_field.u_sorted,
                method=turbine_grid.average_method,
                cubature_weights=turbine_grid.cubature_weights
            )
            # Check: should self.tilt_angle_i be updated? Could it just be saved?
            self.tilt_angle_i = calculate_tilt_for_rotor_effective_velocities(
                turbine_grid_farm,
                average_velocities
            )[:, i:i+1, None, None]

            # Model calculations
            deflection_field_y, deflection_field_z = self.deflection(
                mixing_i,
                thrust_coefficient_i,
                grid.x_sorted,
            )

            velocity_deficit = self.velocity_deficit(
                deflection_field_y,
                deflection_field_z,
                mixing_i,
                thrust_coefficient_i,
                grid.x_sorted,
                grid.y_sorted,
                grid.z_sorted
            )

            wake_field = self.combination_function(
                wake_field,
                velocity_deficit * flow_field.u_initial_sorted
            )

            flow_field.u_sorted = flow_field.u_initial_sorted - wake_field

        flow_field.turbulence_intensity_field_sorted = turbulence_intensity_field


# @profile
def rCalt(wind_veer, sigma_y, sigma_z, y, y_i, delta_y, delta_z, z, HH, Ct,
    yaw, tilt, D, sigma_y0, sigma_z0):

    ## Numexpr
    wind_veer = np.deg2rad(wind_veer)
    a = ne.evaluate(
        "cos(wind_veer) ** 2 / (2 * sigma_y ** 2) + sin(wind_veer) ** 2 / (2 * sigma_z ** 2)"
    )
    b = ne.evaluate(
        "-sin(2 * wind_veer) / (4 * sigma_y ** 2) + sin(2 * wind_veer) / (4 * sigma_z ** 2)"
    )
    c = ne.evaluate(
        "sin(wind_veer) ** 2 / (2 * sigma_y ** 2) + cos(wind_veer) ** 2 / (2 * sigma_z ** 2)"
    )
    r = ne.evaluate(
        "a * ( (y - y_i - delta_y) ** 2) - "+\
        "2 * b * (y - y_i - delta_y) * (z - HH - delta_z) + "+\
        "c * ((z - HH - delta_z) ** 2)"
    )
    d = 1 - Ct * (sigma_y0 * sigma_z0)/(sigma_y * sigma_z) * cosd(yaw) * cosd(tilt)
    C = ne.evaluate("1 - sqrt(d)")
    return r, C

def sigmoid_integral(x, center=0, width=1):
    y = np.zeros_like(x)
    # TODO: Can this be made faster?
    above_smoothing_zone = (x-center) > width/2
    y[above_smoothing_zone] = (x-center)[above_smoothing_zone]
    in_smoothing_zone = ((x-center) >= -width/2) & ((x-center) <= width/2)
    z = ((x-center)/width + 0.5)[in_smoothing_zone]
    if width.shape[0] > 1: # multiple turbine sizes
        width = np.broadcast_to(width, x.shape)[in_smoothing_zone]
    y[in_smoothing_zone] = (width*(z**6 - 3*z**5 + 5/2*z**4)).flatten()
    return y

def empirical_gauss_model_wake_width(
    x,
    wake_expansion_rates,
    breakpoints,
    sigma_0,
    smoothing_length,
    mixing_final,
    ):
    assert len(wake_expansion_rates) == len(breakpoints) + 1, \
        "Invalid combination of wake_expansion_rates and breakpoints."

    sigma = (wake_expansion_rates[0] + mixing_final) * x + sigma_0
    for ib, b in enumerate(breakpoints):
        sigma += (wake_expansion_rates[ib+1] - wake_expansion_rates[ib]) * \
            sigmoid_integral(x, center=b, width=smoothing_length)

    return sigma

def awc_added_wake_mixing(
    awc_mode_i,
    awc_amplitude_i,
    awc_frequency_i,
    awc_wake_exp,
    awc_wake_denominator
):
    # Drop surplus (grid) dimensions
    awc_amplitude_i = awc_amplitude_i[:,:,0,0]
    awc_mode_i = awc_mode_i[:,:,0,0]

    # TODO: Add TI in the mix, finetune amplitude/freq effect
    awc_mixing_factor = np.zeros_like(awc_amplitude_i, dtype=floris_float_type)
    helix_mask = awc_mode_i == 'helix'

    awc_mixing_factor[helix_mask] = (
        awc_amplitude_i[helix_mask]**awc_wake_exp/awc_wake_denominator
    )

    return awc_mixing_factor

def yaw_added_wake_mixing(
    axial_induction_i,
    yaw_angle_i,
    downstream_distance_D_i,
    yaw_added_mixing_gain
):
    return (
        axial_induction_i[:,:,0,0]
        * yaw_added_mixing_gain
        * (1 - cosd(yaw_angle_i[:,:,0,0]))
        / downstream_distance_D_i**2
    )
