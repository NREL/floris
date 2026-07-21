import copy

import numexpr as ne
import numpy as np
from attrs import (
    define,
    field,
    fields,
)
from scipy.special import gamma

from floris.core import (
    axial_induction,
    BaseModel,
    Farm,
    FlowField,
    FlowFieldPlanarGrid,
    PointsGrid,
    thrust_coefficient,
    TurbineGrid,
)
from floris.core.rotor_velocity import (
    average_velocity,
    calculate_tilt_for_rotor_effective_velocities,
)
from floris.core.wake_deflection.gauss import (
    calculate_transverse_velocity,
    wake_added_yaw,
    yaw_added_turbulence_mixing,
)
from floris.core.wake_model import BaseWakeModel
from floris.type_dec import NDArrayFloat
from floris.utilities import cosd, sind, tand


NUM_EPS = fields(BaseModel).NUM_EPS.default


def wake_expansion(
    delta_x,
    ct_i,
    turbulence_intensity_i,
    rotor_diameter,
    a_s,
    b_s,
    c_s1,
    c_s2,
):
    # Calculate Beta (Eq 10, pp 5 of ref. [1] and table 4 of ref. [2] in docstring)
    beta = 0.5 * (1.0 + np.sqrt(1.0 - ct_i)) / np.sqrt(1.0 - ct_i)
    k = a_s * turbulence_intensity_i + b_s
    eps = (c_s1 * ct_i + c_s2) * np.sqrt(beta)

    # Calculate sigma_tilde (Eq 9, pp 5 of ref. [1] and table 4 of ref. [2] in docstring)
    x_tilde = np.abs(delta_x) / rotor_diameter
    sigma_y = k * x_tilde + eps

    return sigma_y


@define
class CumulativeCurl(BaseWakeModel):
    """
    The cumulative curl model is an implementation of the model described in
    :cite:`cc-bay_2022`, which itself is based on the cumulative model of
    :cite:`cc-bastankhah_2021`.

    References:
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: cc-
    """

    # Cumulative Gauss Curl velocity deficit parameters
    a_s: float = field(default=0.179367259)
    b_s: float = field(default=0.0118889215)
    c_s1: float = field(default=0.0563691592)
    c_s2: float = field(default=0.13290157)
    a_f: float = field(default=3.11)
    b_f: float = field(default=-0.68)
    c_f: float = field(default=2.41)
    alpha_mod: float = field(default=1.0)

    # Gauss deflection model parameters
    ad: float = field(converter=float, default=0.0)
    bd: float = field(converter=float, default=0.0)
    alpha: float = field(converter=float, default=0.58)
    beta: float = field(converter=float, default=0.077)
    ka: float = field(converter=float, default=0.38)
    kb: float = field(converter=float, default=0.004)
    dm: float = field(converter=float, default=1.0)
    eps_gain: float = field(converter=float, default=0.2)
    use_secondary_steering: bool = field(converter=bool, default=True)

    # Crespo-Hernandez turbulence model parameters
    initial: float = field(converter=float, default=0.1)
    constant: float = field(converter=float, default=0.9)
    ai: float = field(converter=float, default=0.8)
    downstream: float = field(converter=float, default=-0.32)

    # Secondary effects parameters
    enable_secondary_steering: bool = field(converter=bool, default=True)
    enable_transverse_velocities: bool = field(converter=bool, default=True)
    enable_yaw_added_recovery: bool = field(converter=bool, default=True)

    # Instance variables (not initialized)
    effective_yaw_i: np.ndarray = field(init=False, default=None)
    ambient_turbulence_intensities: np.ndarray = field(init=False, default=None)
    wind_veer: float = field(init=False, default=None)
    freestream_velocity: np.ndarray = field(init=False, default=None)
    turb_u_wake: np.ndarray = field(init=False, default=None)
    Ctmp: np.ndarray = field(init=False, default=None)
    turb_inflow_field: np.ndarray = field(init=False, default=None)
    turb_Cts: np.ndarray = field(init=False, default=None)

    def velocity_deficit(
        self,
        ii: int,
        u_i: np.ndarray,
        deflection_field: np.ndarray,
        turbulence_intensity: np.ndarray,
        ct: np.ndarray,
        turbine_diameter: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        u_initial: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Cumulative velocity deficit calculation. Updates and returns turb_u_wake and Ctmp.
        """
        turbine_Ct = ct
        turbine_ti = turbulence_intensity
        turbine_yaw = self.yaw_angle_i

        # Cubic mean velocity at current turbine
        turb_avg_vels = np.cbrt(np.mean(u_i ** 3, axis=(2, 3), keepdims=True))

        delta_x = x - self.x_i

        sigma_n = wake_expansion(
            delta_x,
            turbine_Ct[:, ii:ii+1],
            turbine_ti[:, ii:ii+1],
            turbine_diameter[:, ii:ii+1],
            self.a_s,
            self.b_s,
            self.c_s1,
            self.c_s2,
        )

        y_i_loc = np.mean(self.y_i, axis=(2, 3), keepdims=True)
        z_i_loc = np.mean(self.z_i, axis=(2, 3), keepdims=True)

        x_coord = np.mean(x, axis=(2, 3), keepdims=True)
        y_coord = np.mean(y, axis=(2, 3), keepdims=True)
        z_coord = np.mean(z, axis=(2, 3), keepdims=True)

        sum_lbda = np.zeros_like(u_initial)

        # Cumulative effects from all upstream turbines
        for m in range(0, ii - 1):
            x_coord_m = x_coord[:, m:m+1]
            y_coord_m = y_coord[:, m:m+1]
            z_coord_m = z_coord[:, m:m+1]

            if x_coord[:, m:m+1].size == 0:
                break

            delta_x_m = x - x_coord_m

            sigma_i = wake_expansion(
                delta_x_m,
                turbine_Ct[:, m:m+1],
                turbine_ti[:, m:m+1],
                turbine_diameter[:, m:m+1],
                self.a_s,
                self.b_s,
                self.c_s1,
                self.c_s2,
            )

            S_i = sigma_n ** 2 + sigma_i ** 2

            Y_i = (y_i_loc - y_coord_m - deflection_field) ** 2 / (2 * S_i)
            Z_i = (z_i_loc - z_coord_m) ** 2 / (2 * S_i)

            lbda = 1.0 * sigma_i ** 2 / S_i * np.exp(-Y_i) * np.exp(-Z_i)

            sum_lbda = sum_lbda + lbda * (self.Ctmp[m] / u_initial)

        # Super-Gaussian velocity deficit (Blondel model with cumulative effects)
        x_tilde = np.abs(delta_x) / turbine_diameter[:, ii:ii+1]
        r_tilde = np.sqrt(
            (y - y_i_loc - deflection_field) ** 2 + (z - z_i_loc) ** 2
        )
        r_tilde /= turbine_diameter[:, ii:ii+1]

        n = self.a_f * np.exp(self.b_f * x_tilde) + self.c_f
        a1 = 2 ** (2 / n - 1)
        a2 = 2 ** (4 / n - 2)

        # Blondel model with cumulative effects
        tmp = a2 - (
            (n * turbine_Ct[:, ii:ii+1])
            * cosd(turbine_yaw)
            / (
                16.0
                * gamma(2 / n)
                * np.sign(sigma_n)
                * (np.abs(sigma_n) ** (4 / n))
                * (1 - sum_lbda) ** 2
            )
        )

        # Replace negative values with zeros to prevent NaNs
        tmp = tmp * (tmp >= 0)

        C = a1 - np.sqrt(tmp)
        C = C * (1 - sum_lbda)

        self.Ctmp[ii] = C

        yR = y - y_i_loc
        xR = yR * tand(turbine_yaw) + self.x_i

        # Velocity deficit
        velDef = C * np.exp((-1 * r_tilde ** n) / (2 * sigma_n ** 2))
        velDef = velDef * (x - xR >= 0.1)

        self.turb_u_wake = self.turb_u_wake + turb_avg_vels * velDef
        return (self.turb_u_wake, self.Ctmp)

    def deflection(
        self,
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        x: np.ndarray,
        x_i: np.ndarray | None = None,
        freestream_velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Gauss deflection model.
        """
        # Use provided x_i or fall back to instance variable (set in turbine_solve)
        if x_i is None:
            x_i = self.x_i
        if freestream_velocity is None:
            freestream_velocity = self.freestream_velocity
        
        # Opposite sign convention
        yaw_i = -1 * self.effective_yaw_i

        # initial velocity deficits
        uR = (
            freestream_velocity
            * ct_i
            * cosd(0.0)
            * cosd(yaw_i)
            / (2.0 * (1 - np.sqrt(1 - (ct_i * cosd(0.0) * cosd(yaw_i)))))
        )
        u0 = freestream_velocity * np.sqrt(1 - ct_i)

        # length of near wake
        x0 = (
            self.rotor_diameter_i
            * (cosd(yaw_i) * (1 + np.sqrt(1 - ct_i * cosd(yaw_i))))
            / (np.sqrt(2) * (
                4 * self.alpha * turbulence_intensity_i + 2 * self.beta * (1 - np.sqrt(1 - ct_i))
            )) + x_i
        )

        # wake expansion parameters
        ky = self.ka * turbulence_intensity_i + self.kb
        kz = self.ka * turbulence_intensity_i + self.kb

        C0 = 1 - u0 / freestream_velocity
        M0 = C0 * (2 - C0)
        E0 = ne.evaluate("C0 ** 2 - 3 * exp(1.0 / 12.0) * C0 + 3 * exp(1.0 / 3.0)")

        # initial Gaussian wake expansion
        freestream_velocity_local = freestream_velocity
        rotor_diameter_i = self.rotor_diameter_i
        sigma_z0 = ne.evaluate("rotor_diameter_i * 0.5 * sqrt(uR / (freestream_velocity_local + u0))")
        sigma_y0 = sigma_z0 * cosd(yaw_i) * cosd(self.wind_veer)

        xR = x_i

        # yaw parameters (skew angle and distance from centerline)
        theta_c0 = self.dm * (0.3 * np.radians(yaw_i) / cosd(yaw_i))
        theta_c0 *= (1 - np.sqrt(1 - ct_i * cosd(yaw_i)))
        delta0 = np.tan(theta_c0) * (x0 - x_i)

        # deflection in the near wake
        delta_near_wake = ((x - xR) / (x0 - xR)) * delta0 + (self.ad + self.bd * (x - x_i))
        delta_near_wake *= (x >= xR) & (x <= x0)

        # deflection in the far wake
        sigma_y = ky * (x - x0) + sigma_y0
        sigma_z = kz * (x - x0) + sigma_z0
        sigma_y = sigma_y * (x >= x0) + sigma_y0 * (x < x0)
        sigma_z = sigma_z * (x >= x0) + sigma_z0 * (x < x0)

        M0_sqrt = np.sqrt(M0)
        middle_term = np.sqrt(sigma_y * sigma_z / (sigma_y0 * sigma_z0))
        ln_deltaNum = (1.6 + M0_sqrt) * (1.6 * middle_term - M0_sqrt)
        ln_deltaDen = (1.6 - M0_sqrt) * (1.6 * middle_term + M0_sqrt)

        middle_term = ne.evaluate(
            "theta_c0"
            " * E0"
            " / 5.2"
            " * sqrt(sigma_y0 * sigma_z0 / (ky * kz * M0))"
            " * log(ln_deltaNum / ln_deltaDen)"
        )
        delta_far_wake = delta0 + middle_term + (self.ad + self.bd * (x - x_i))

        delta_far_wake = delta_far_wake * (x > x0)
        deflection = delta_near_wake + delta_far_wake

        return deflection

    def turbulence(
        self,
        turbulence_intensity: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        axial_induction: np.ndarray,
        area_overlap: np.ndarray,
    ) -> np.ndarray:
        """
        Crespo-Hernandez turbulence model.
        """
        x_i = self.x_i
        rotor_diameter_i = self.rotor_diameter_i
        delta_x = x - x_i
        ambient_TI = self.ambient_turbulence_intensities

        upstream_mask = delta_x <= 0.1
        downstream_mask = delta_x > -0.1

        delta_x = delta_x * downstream_mask + np.ones_like(delta_x) * upstream_mask

        # Crespo et al. turbulence intensity calculation
        constant = self.constant
        ai = self.ai
        initial = self.initial
        downstream = self.downstream
        ti = ne.evaluate(
            "constant"
            " * axial_induction ** ai"
            " * ambient_TI ** initial"
            " * (delta_x / rotor_diameter_i) ** downstream"
        )
        wake_added_turbulence_intensity = ti * downstream_mask

        # Modify wake added turbulence by wake area overlap
        downstream_influence_length = 15 * self.rotor_diameter_i
        ti_added = (
            area_overlap
            * np.nan_to_num(wake_added_turbulence_intensity, posinf=0.0)
            * (x > self.x_i)
            * (np.abs(self.y_i - y) < 2 * self.rotor_diameter_i)
            * (x <= downstream_influence_length + self.x_i)
        )

        # Combine turbine TIs with WAT
        turbulence_intensity = np.maximum(
            np.sqrt(ti_added**2 + ambient_TI**2), turbulence_intensity
        )

        return turbulence_intensity

    def turbine_solve(
        self,
        farm: Farm,
        flow_field: FlowField,
        grid: TurbineGrid,
    ) -> None:
        """
        Solve for turbines using the cumulative curl model.
        """
        # Initialize wake state
        v_wake = np.zeros_like(flow_field.v_initial_sorted)
        w_wake = np.zeros_like(flow_field.w_initial_sorted)
        self.turb_u_wake = np.zeros_like(flow_field.u_initial_sorted)
        self.turb_inflow_field = copy.deepcopy(flow_field.u_initial_sorted)

        # Set up turbulence arrays
        turbine_turbulence_intensity = flow_field.turbulence_intensities[:, None, None, None]
        turbine_turbulence_intensity = np.repeat(turbine_turbulence_intensity, farm.n_turbines, axis=1)

        # Ambient turbulent intensity
        self.ambient_turbulence_intensities = flow_field.turbulence_intensities.copy()
        self.ambient_turbulence_intensities = self.ambient_turbulence_intensities[:, None, None, None]

        # Initialize state arrays for cumulative calculation
        shape = (farm.n_turbines,) + np.shape(flow_field.u_initial_sorted)
        self.Ctmp = np.zeros((shape))

        # Copy uniform flow field parameters
        self.freestream_velocity = flow_field.u_initial_sorted
        self.wind_veer = flow_field.wind_veer

        # Calculate the velocity deficit sequentially from upstream to downstream turbines
        for i in range(grid.n_turbines):

            # Get the current turbine quantities
            self.set_turbine_i(grid, farm, i)

            # Compute rotor-vicinity mask for inflow field update
            rotor_diameter_i = farm.rotor_diameters_sorted[:, i:i+1, None, None]
            mask2 = (
                (grid.x_sorted < self.x_i + 0.01)
                * (grid.x_sorted > self.x_i - 0.01)
                * (grid.y_sorted < self.y_i + 0.51 * rotor_diameter_i)
                * (grid.y_sorted > self.y_i - 0.51 * rotor_diameter_i)
            )
            self.turb_inflow_field = (
                self.turb_inflow_field * ~mask2
                + (flow_field.u_initial_sorted - self.turb_u_wake) * mask2
            )

            # Compute thrust coefficients for all turbines using turbine inflow field
            turb_avg_vels = average_velocity(self.turb_inflow_field)[:, :, None, None]
            self.turb_Cts = thrust_coefficient(
                turbines=farm.turbines,
                velocities=turb_avg_vels,
                turbulence_intensities=flow_field.turbulence_intensity_field_sorted,
                air_density=flow_field.air_density,
                yaw_angles=farm.yaw_angles_sorted,
                power_setpoints=farm.power_setpoints_sorted,
                awc_modes=farm.awc_modes_sorted,
                awc_amplitudes=farm.awc_amplitudes_sorted,
                turbine_type_map=farm.turbine_type_map_sorted,
                average_method=grid.average_method,
                cubature_weights=grid.cubature_weights,
                multidim_condition=flow_field.multidim_conditions,
            )
            self.turb_Cts = self.turb_Cts[:, :, None, None]

            # Compute axial induction for current turbine
            turb_aIs = axial_induction(
                turbines=farm.turbines,
                velocities=turb_avg_vels,
                turbulence_intensities=flow_field.turbulence_intensity_field_sorted,
                air_density=flow_field.air_density,
                yaw_angles=farm.yaw_angles_sorted,
                power_setpoints=farm.power_setpoints_sorted,
                awc_modes=farm.awc_modes_sorted,
                awc_amplitudes=farm.awc_amplitudes_sorted,
                turbine_type_map=farm.turbine_type_map_sorted,
                ix_filter=[i],
                average_method=grid.average_method,
                cubature_weights=grid.cubature_weights,
                multidim_condition=flow_field.multidim_conditions,
            )
            turb_aIs = turb_aIs[:, :, None, None]

            u_i = self.turb_inflow_field[:, i:i+1]
            v_i = flow_field.v_sorted[:, i:i+1]

            # Axial induction for current turbine
            axial_induction_i = axial_induction(
                turbines=farm.turbines,
                velocities=flow_field.u_sorted,
                turbulence_intensities=flow_field.turbulence_intensity_field_sorted,
                air_density=flow_field.air_density,
                yaw_angles=farm.yaw_angles_sorted,
                power_setpoints=farm.power_setpoints_sorted,
                awc_modes=farm.awc_modes_sorted,
                awc_amplitudes=farm.awc_amplitudes_sorted,
                turbine_type_map=farm.turbine_type_map_sorted,
                ix_filter=[i],
                average_method=grid.average_method,
                cubature_weights=grid.cubature_weights,
                multidim_condition=flow_field.multidim_conditions,
            )
            axial_induction_i = axial_induction_i[:, :, None, None]

            turbulence_intensity_i = turbine_turbulence_intensity[:, i:i+1]
            yaw_angle_i = farm.yaw_angles_sorted[:, i:i+1, None, None]
            hub_height_i = farm.hub_heights_sorted[:, i:i+1, None, None]
            TSR_i = farm.TSRs_sorted[:, i:i+1, None, None]

            # Initialize effective yaw angle
            self.effective_yaw_i = yaw_angle_i.copy()

            if self.enable_secondary_steering:
                added_yaw = wake_added_yaw(
                    u_i,
                    v_i,
                    flow_field.u_initial_sorted,
                    grid.y_sorted[:, i:i+1] - self.y_i,
                    grid.z_sorted[:, i:i+1],
                    self.rotor_diameter_i,
                    hub_height_i,
                    self.turb_Cts[:, i:i+1],
                    TSR_i,
                    axial_induction_i,
                    flow_field.wind_shear,
                    scale=2.0,
                )
                self.effective_yaw_i += added_yaw

            # Compute deflection
            deflection_field = self.deflection(
                turbulence_intensity_i,
                self.turb_Cts[:, i:i+1],
                grid.x_sorted,
            )

            if self.enable_transverse_velocities:
                v_wake, w_wake = calculate_transverse_velocity(
                    u_i,
                    flow_field.u_initial_sorted,
                    flow_field.dudz_initial_sorted,
                    grid.x_sorted - self.x_i,
                    grid.y_sorted - self.y_i,
                    grid.z_sorted,
                    self.rotor_diameter_i,
                    hub_height_i,
                    yaw_angle_i,
                    self.turb_Cts[:, i:i+1],
                    TSR_i,
                    axial_induction_i,
                    flow_field.wind_shear,
                    scale=2.0,
                )

            if self.enable_yaw_added_recovery:
                I_mixing = yaw_added_turbulence_mixing(
                    u_i,
                    turbulence_intensity_i,
                    v_i,
                    flow_field.w_sorted[:, i:i+1],
                    v_wake[:, i:i+1],
                    w_wake[:, i:i+1],
                )
                gch_gain = 1.0
                turbine_turbulence_intensity[:, i:i+1] = turbulence_intensity_i + gch_gain * I_mixing

            # Compute velocity deficit (cumulative)
            self.turb_u_wake, self.Ctmp = self.velocity_deficit(
                i,
                u_i,
                deflection_field,
                turbine_turbulence_intensity,
                self.turb_Cts,
                farm.rotor_diameters_sorted[:, :, None, None],
                grid.x_sorted,
                grid.y_sorted,
                grid.z_sorted,
                flow_field.u_initial_sorted,
            )

            # Calculate wake overlap for wake-added turbulence (WAT)
            area_overlap = 1 - (
                np.sum(self.turb_u_wake <= 0.05, axis=(2, 3), keepdims=True)
                / (grid.grid_resolution * grid.grid_resolution)
            )

            # Compute wake-added turbulence with area overlap
            wake_added_turbulence_intensity = self.turbulence(
                self.ambient_turbulence_intensities,
                grid.x_sorted,
                grid.y_sorted,
                turb_aIs,
                area_overlap,
            )

            # Combine turbine TIs with WAT
            turbine_turbulence_intensity = np.maximum(
                wake_added_turbulence_intensity,
                turbine_turbulence_intensity
            )

            flow_field.v_sorted += v_wake
            flow_field.w_sorted += w_wake

        flow_field.u_sorted = self.turb_inflow_field
        flow_field.turbulence_intensity_field_sorted = turbine_turbulence_intensity
        flow_field.turbulence_intensity_field_sorted_avg = np.mean(
            turbine_turbulence_intensity,
            axis=(2, 3),
            keepdims=True
        )

    def point_solve(
        self,
        farm: Farm,
        flow_field: FlowField,
        grid: FlowFieldPlanarGrid | PointsGrid,
    ) -> None:
        """
        Solve for a general point grid using the cumulative curl model.
        Mimics full_flow_cc_solver from solver.py.
        """
        # Get the flow quantities and turbine performance on turbine grid
        turbine_grid_farm = copy.deepcopy(farm)
        turbine_grid_flow_field = copy.deepcopy(flow_field)

        turbine_grid = TurbineGrid(
            turbine_coordinates=turbine_grid_farm.coordinates,
            turbine_diameters=turbine_grid_farm.rotor_diameters,
            wind_directions=turbine_grid_flow_field.wind_directions,
            grid_resolution=3,
        )
        turbine_grid_farm.set_sorted_indices(turbine_grid.sorted_coord_indices)
        turbine_grid_farm.construct_turbine_type_map()
        turbine_grid_flow_field.initialize_velocity_field(turbine_grid)
        turbine_grid_farm.initialize()

        # Run turbine solve to populate state
        self.turbine_solve(turbine_grid_farm, turbine_grid_flow_field, turbine_grid)

        # Now compute wake field on the full grid
        v_wake = np.zeros_like(flow_field.v_initial_sorted)
        w_wake = np.zeros_like(flow_field.w_initial_sorted)
        turb_u_wake = np.zeros_like(flow_field.u_initial_sorted)

        # Initialize the turbulence intensity field over the entire flow field grid
        n_points = grid.x_sorted.shape[1]
        ambient_turbulence_intensities = flow_field.turbulence_intensities[:, None, None, None]
        ambient_turbulence_intensities = np.repeat(ambient_turbulence_intensities, n_points, axis=1)
        turbulence_intensity_field = ambient_turbulence_intensities.copy()

        shape = (farm.n_turbines,) + np.shape(flow_field.u_initial_sorted)
        Ctmp = np.zeros((shape))

        # Calculate the velocity deficit sequentially from upstream to downstream turbines
        for i in range(grid.n_turbines):

            # Set self.Ctmp and self.turb_u_wake for this iteration (point_solve uses local versions for full grid)
            self.Ctmp = Ctmp
            self.turb_u_wake = turb_u_wake

            # Get the current turbine quantities
            self.set_turbine_i(turbine_grid, turbine_grid_farm, i)

            u_i = turbine_grid_flow_field.u_sorted[:, i:i+1]
            v_i = turbine_grid_flow_field.v_sorted[:, i:i+1]

            # Use saved turb_Cts from turbine_solve
            turb_Cts_i = self.turb_Cts

            # Axial induction
            axial_induction_i = axial_induction(
                turbines=farm.turbines,
                velocities=turbine_grid_flow_field.u_sorted,
                turbulence_intensities=turbine_grid_flow_field.turbulence_intensity_field_sorted,
                air_density=turbine_grid_flow_field.air_density,
                yaw_angles=turbine_grid_farm.yaw_angles_sorted,
                power_setpoints=turbine_grid_farm.power_setpoints_sorted,
                awc_modes=turbine_grid_farm.awc_modes_sorted,
                awc_amplitudes=turbine_grid_farm.awc_amplitudes_sorted,
                turbine_type_map=turbine_grid_farm.turbine_type_map_sorted,
                ix_filter=[i],
                average_method=turbine_grid.average_method,
                cubature_weights=turbine_grid.cubature_weights,
                multidim_condition=turbine_grid_flow_field.multidim_conditions,
            )
            axial_induction_i = axial_induction_i[:, :, None, None]

            turbulence_intensity_i = \
                turbine_grid_flow_field.turbulence_intensity_field_sorted_avg[:, i:i+1]
            yaw_angle_i = turbine_grid_farm.yaw_angles_sorted[:, i:i+1, None, None]
            hub_height_i = turbine_grid_farm.hub_heights_sorted[:, i:i+1, None, None]
            TSR_i = turbine_grid_farm.TSRs_sorted[:, i:i+1, None, None]

            self.effective_yaw_i = yaw_angle_i.copy()

            if self.enable_secondary_steering:
                added_yaw = wake_added_yaw(
                    u_i,
                    v_i,
                    turbine_grid_flow_field.u_initial_sorted,
                    turbine_grid.y_sorted[:, i:i+1] - self.y_i,
                    turbine_grid.z_sorted[:, i:i+1],
                    self.rotor_diameter_i,
                    hub_height_i,
                    turb_Cts_i[:, i:i+1],
                    TSR_i,
                    axial_induction_i,
                    flow_field.wind_shear,
                    scale=2.0,
                )
                self.effective_yaw_i += added_yaw

            # Model calculations
            deflection_field = self.deflection(
                turbulence_intensity_i,
                turb_Cts_i[:, i:i+1],
                grid.x_sorted,
                self.x_i,
                flow_field.u_initial_sorted,
            )

            if self.enable_transverse_velocities:
                v_wake, w_wake = calculate_transverse_velocity(
                    u_i,
                    flow_field.u_initial_sorted,
                    flow_field.dudz_initial_sorted,
                    grid.x_sorted - self.x_i,
                    grid.y_sorted - self.y_i,
                    grid.z_sorted,
                    self.rotor_diameter_i,
                    hub_height_i,
                    yaw_angle_i,
                    turb_Cts_i[:, i:i+1],
                    TSR_i,
                    axial_induction_i,
                    flow_field.wind_shear,
                    scale=2.0,
                )

            # Velocity deficit (cumulative)
            turb_u_wake, Ctmp = self.velocity_deficit(
                i,
                u_i,
                deflection_field,
                turbine_grid_flow_field.turbulence_intensity_field_sorted_avg,
                turb_Cts_i,
                turbine_grid_farm.rotor_diameters_sorted[:, :, None, None],
                grid.x_sorted,
                grid.y_sorted,
                grid.z_sorted,
                flow_field.u_initial_sorted,
            )

            # Calculate wake overlap for wake-added turbulence (WAT)
            area_overlap = np.where(turb_u_wake > 0.05, 1, 0)

            # Compute wake-added turbulence with area overlap
            wake_added_turbulence_intensity = self.turbulence(
                ambient_turbulence_intensities,
                grid.x_sorted,
                grid.y_sorted,
                axial_induction_i,
                area_overlap,
            )

            # Combine turbine TIs with WAT
            turbulence_intensity_field = np.maximum(
                wake_added_turbulence_intensity,
                turbulence_intensity_field
            )

            flow_field.v_sorted += v_wake
            flow_field.w_sorted += w_wake

        flow_field.u_sorted = flow_field.u_initial_sorted - turb_u_wake
        flow_field.turbulence_intensity_field_sorted = turbulence_intensity_field
