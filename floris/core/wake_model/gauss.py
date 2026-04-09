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
    FlowFieldGrid,
    FlowFieldPlanarGrid,
    PointsGrid,
    TurbineGrid,
)
from floris.core.wake_deflection.gauss import (
    calculate_transverse_velocity,
    wake_added_yaw,
    yaw_added_turbulence_mixing,
)
from floris.core.wake_model import BaseWakeModel
from floris.utilities import cosd


NUM_EPS = fields(BaseModel).NUM_EPS.default

@define
class Gauss(BaseWakeModel):

    # Gauss deficit model parameters
    alpha: float = field(default=0.58)
    beta: float = field(default=0.077)
    ka: float = field(default=0.38)
    kb: float = field(default=0.004)

    # Gauss deflection model parameters
    ad: float = field(converter=float, default=0.0)
    bd: float = field(converter=float, default=0.0)
    dm: float = field(converter=float, default=1.0)
    eps_gain: float = field(converter=float, default=0.2)
    use_secondary_steering: bool = field(converter=bool, default=True)

    # Crespo-Hernandez turbulence model parameters
    initial: float = field(converter=float, default=0.1)
    constant: float = field(converter=float, default=0.9)
    ai: float = field(converter=float, default=0.8)
    downstream: float = field(converter=float, default=-0.32)

    # Secondary effects parameters (GCH)
    enable_transverse_velocities: bool = field(converter=bool, default=True)
    enable_yaw_added_recovery: bool = field(converter=bool, default=True)
    enable_secondary_steering: bool = field(converter=bool, default=True)

    effective_yaw_i: np.ndarray = field(init=False)

    ambient_turbulence_intensities: np.ndarray = field(init=False)
    wind_veer: float = field(init=False)
    freestream_velocity: np.ndarray = field(init=False)

    def velocity_deficit(
        self,
        axial_induction_i: np.ndarray,
        deflection_field_i: np.ndarray,
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:

        # yaw_angle is all turbine yaw angles for each wind speed
        # Extract and broadcast only the current turbine yaw setting
        # for all wind speeds

        # Opposite sign convention in this model
        yaw_angle = -1 * self.yaw_angle_i

        # Initialize the velocity deficit
        uR = self.freestream_velocity * ct_i / (2.0 * (1 - np.sqrt(1 - ct_i)))
        u0 = self.freestream_velocity * np.sqrt(1 - ct_i)

        # Initial lateral bounds
        sigma_z0 = self.rotor_diameter_i * 0.5 * np.sqrt(uR / (self.freestream_velocity + u0))
        sigma_y0 = sigma_z0 * cosd(yaw_angle) * cosd(self.wind_veer)

        # Compute the bounds of the near and far wake regions and a mask

        # Start of the near wake
        xR = self.x_i

        # Start of the far wake
        x0 = np.ones_like(self.freestream_velocity)
        x0 *= self.rotor_diameter_i * cosd(yaw_angle) * (1 + np.sqrt(1 - ct_i) )
        x0 /= np.sqrt(2) * (
            4 * self.alpha * turbulence_intensity_i + 2 * self.beta * (1 - np.sqrt(1 - ct_i) )
        )
        x0 += self.x_i

        # Initialize the velocity deficit array
        velocity_deficit = np.zeros_like(self.freestream_velocity)

        # Masks
        # When we have only an inequality, the current turbine may be applied its own
        # wake in cases where numerical precision cause in incorrect comparison. We've
        # applied a small bump to avoid this. "0.1" is arbitrary but it is a small, non
        # zero value.

        # This mask defines the near wake; keeps the areas downstream of xR and upstream of x0
        near_wake_mask = (x > xR + 0.1) * (x < x0)
        far_wake_mask = (x >= x0)

        # Compute the velocity deficit in the NEAR WAKE region
        # ONLY If there are points within the near wake boundary
        # TODO: for the TurbineGrid, do we need to do this near wake calculation at all?
        #       same question for any grid with a resolution larger than the near wake region
        if np.sum(near_wake_mask):

            # Calculate the wake expansion

            # This is a linear ramp from 0 to 1 from the start of the near wake to the start
            # of the far wake.
            near_wake_ramp_up = (x - xR) / (x0 - xR)
            # Another linear ramp, but positive upstream of the far wake and negative in the
            # far wake; 0 at the start of the far wake
            near_wake_ramp_down = (x0 - x) / (x0 - xR)
            # near_wake_ramp_down = -1 * (near_wake_ramp_up - 1)  # : this is equivalent, right?

            sigma_y = near_wake_ramp_down * 0.501 * self.rotor_diameter_i * np.sqrt(ct_i / 2.0)
            sigma_y += near_wake_ramp_up * sigma_y0
            sigma_y *= (x >= xR)
            sigma_y += np.ones_like(sigma_y) * (x < xR) * 0.5 * self.rotor_diameter_i

            sigma_z = near_wake_ramp_down * 0.501 * self.rotor_diameter_i * np.sqrt(ct_i / 2.0)
            sigma_z += near_wake_ramp_up * sigma_z0
            sigma_z *= (x >= xR)
            sigma_z += np.ones_like(sigma_z) * (x < xR) * 0.5 * self.rotor_diameter_i

            r_squared, C = rC(
                self.wind_veer,
                sigma_y,
                sigma_z,
                y,
                self.y_i,
                deflection_field_i,
                z,
                self.hub_height_i,
                ct_i,
                yaw_angle,
                self.rotor_diameter_i,
            )

            near_wake_deficit = gaussian_function(C, r_squared, 1, np.sqrt(0.5))
            near_wake_deficit *= near_wake_mask

            velocity_deficit += near_wake_deficit

        # Compute the velocity deficit in the FAR WAKE region
        if np.sum(far_wake_mask):

            # Wake expansion in the lateral (y) and the vertical (z)
            ky = self.ka * turbulence_intensity_i + self.kb  # wake expansion parameters
            kz = self.ka * turbulence_intensity_i + self.kb  # wake expansion parameters
            sigma_y = (ky * (x - x0) + sigma_y0) * far_wake_mask + sigma_y0 * (x < x0)
            sigma_z = (kz * (x - x0) + sigma_z0) * far_wake_mask + sigma_z0 * (x < x0)

            r_squared, C = rC(
                self.wind_veer,
                sigma_y,
                sigma_z,
                y,
                self.y_i,
                deflection_field_i,
                z,
                self.hub_height_i,
                ct_i,
                yaw_angle,
                self.rotor_diameter_i,
            )

            far_wake_deficit = gaussian_function(C, r_squared, 1, np.sqrt(0.5))
            far_wake_deficit *= far_wake_mask

            velocity_deficit += far_wake_deficit

        return velocity_deficit

    def deflection(
        self,
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the deflection field of the wake. See
        :cite:`gdm-bastankhah2016experimental` and :cite:`gdm-King2019Controls`
        for details on the methods used.

        Args:
            x_i (np.array): x-coordinates of turbine i.
            y_i (np.array): y-coordinates of turbine i.
            yaw_i (np.array): Yaw angle of turbine i.
            turbulence_intensity_i (np.array): Turbulence intensity at turbine i.
            ct_i (np.array): Thrust coefficient of turbine i.
            rotor_diameter_i (float): Rotor diameter of turbine i.

        Returns:
            np.array: Deflection field for the wake.
        """
        # ==============================================================

        # Opposite sign convention in this model
        yaw_i = -1 * self.effective_yaw_i

        # TODO: connect support for tilt
        tilt = 0.0  # turbine.tilt_angle

        # initial velocity deficits
        uR = (
            self.freestream_velocity
            * ct_i
            * cosd(tilt)
            * cosd(yaw_i)
            / (2.0 * (1 - np.sqrt(1 - (ct_i * cosd(tilt) * cosd(yaw_i)))))
        )
        u0 = self.freestream_velocity * np.sqrt(1 - ct_i)

        # length of near wake
        x0 = (
            self.rotor_diameter_i
            * (cosd(yaw_i) * (1 + np.sqrt(1 - ct_i * cosd(yaw_i))))
            / (np.sqrt(2) * (
                4 * self.alpha * turbulence_intensity_i + 2 * self.beta * (1 - np.sqrt(1 - ct_i))
            )) + self.x_i
        )

        # wake expansion parameters
        ky = self.ka * turbulence_intensity_i + self.kb
        kz = self.ka * turbulence_intensity_i + self.kb

        C0 = 1 - u0 / self.freestream_velocity
        M0 = C0 * (2 - C0)
        E0 = ne.evaluate("C0 ** 2 - 3 * exp(1.0 / 12.0) * C0 + 3 * exp(1.0 / 3.0)")

        # initial Gaussian wake expansion
        freestream_velocity = self.freestream_velocity # Extract for numexpr
        rotor_diameter_i = self.rotor_diameter_i # Extract for numexpr
        sigma_z0 = ne.evaluate("rotor_diameter_i * 0.5 * sqrt(uR / (freestream_velocity + u0))")
        sigma_y0 = sigma_z0 * cosd(yaw_i) * cosd(self.wind_veer)

        # yR = y - y_i
        xR = self.x_i # yR * tand(yaw) + x_i

        # yaw parameters (skew angle and distance from centerline)
        # skew angle in radians
        theta_c0 = self.dm * (0.3 * np.radians(yaw_i) / cosd(yaw_i))
        theta_c0 *= (1 - np.sqrt(1 - ct_i * cosd(yaw_i)))
        delta0 = np.tan(theta_c0) * (x0 - self.x_i)  # initial wake deflection;
        # NOTE: use np.tan here since theta_c0 is radians

        # deflection in the near wake
        delta_near_wake = ((x - xR) / (x0 - xR)) * delta0 + (self.ad + self.bd * (x - self.x_i))
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
        delta_far_wake = delta0 + middle_term + (self.ad + self.bd * (x - self.x_i))

        delta_far_wake = delta_far_wake * (x > x0)
        deflection = delta_near_wake + delta_far_wake

        return deflection

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

    def turbulence(
        self,
        turbulence_intensity: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        axial_induction: np.ndarray,
        area_overlap: np.ndarray,
    ) -> np.ndarray:
        # Replace zeros and negatives with 1 to prevent nans/infs
        x_i = self.x_i
        rotor_diameter_i = self.rotor_diameter_i
        delta_x = x - x_i
        ambient_TI = self.ambient_turbulence_intensities

        # TODO: ensure that these fudge factors are needed for different rotations
        upstream_mask = delta_x <= 0.1
        downstream_mask = delta_x > -0.1

        #        Keep downstream components          Set upstream to 1.0
        delta_x = delta_x * downstream_mask + np.ones_like(delta_x) * upstream_mask

        # turbulence intensity calculation based on Crespo et. al.
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
        # Mask the 1 values from above with zeros
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

        wake_field = np.zeros_like(flow_field.u_initial_sorted)

        # Expand input turbulence intensity to 4d for (n_turbines, grid, grid)
        turbine_turbulence_intensity = np.repeat(
            flow_field.turbulence_intensities[:, None, None, None],
            farm.n_turbines,
            axis=1
        )

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
            thrust_coefficient_i = self.turbine_thrust_coefficient(grid, farm, flow_field, i)
            axial_induction_i = self.turbine_axial_induction(grid, farm, flow_field, i)
            u_i = flow_field.u_sorted[:, i:i+1]
            v_i = flow_field.v_sorted[:, i:i+1]
            turbulence_intensity_i = turbine_turbulence_intensity[:, i:i+1]

            # Initialize the effective yaw angle
            self.effective_yaw_i = self.yaw_angle_i.copy()

            # Model calculations
            if self.enable_secondary_steering:
                added_yaw = wake_added_yaw(
                    u_i,
                    v_i,
                    flow_field.u_initial_sorted,
                    grid.y_sorted[:, i:i+1] - self.y_i,
                    grid.z_sorted[:, i:i+1],
                    self.rotor_diameter_i,
                    self.hub_height_i,
                    thrust_coefficient_i,
                    self.TSR_i,
                    axial_induction_i,
                    flow_field.wind_shear,
                )
                self.effective_yaw_i += added_yaw

            deflection_field = self.deflection(
                turbine_turbulence_intensity[:, i:i+1],
                thrust_coefficient_i,
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
                    self.hub_height_i,
                    self.yaw_angle_i,
                    thrust_coefficient_i,
                    self.TSR_i,
                    axial_induction_i,
                    flow_field.wind_shear,
                )
            else:
                v_wake = np.zeros_like(flow_field.v_initial_sorted)
                w_wake = np.zeros_like(flow_field.w_initial_sorted)

            if self.enable_yaw_added_recovery:
                I_mixing = yaw_added_turbulence_mixing(
                    u_i,
                    turbulence_intensity_i,
                    v_i,
                    flow_field.w_sorted[:, i:i+1],
                    v_wake[:, i:i+1],
                    w_wake[:, i:i+1],
                )
                gch_gain = 2
                turbine_turbulence_intensity[:, i:i+1] = (
                    turbulence_intensity_i + gch_gain * I_mixing
                )

            velocity_deficit = self.velocity_deficit(
                axial_induction_i,
                deflection_field,
                turbine_turbulence_intensity[:, i:i+1],
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

            turbine_turbulence_intensity = self.turbulence(
                turbine_turbulence_intensity,
                grid.x_sorted,
                grid.y_sorted,
                axial_induction_i,
                area_overlap,
            )

            flow_field.u_sorted = flow_field.u_initial_sorted - wake_field
            flow_field.v_sorted += v_wake
            flow_field.w_sorted += w_wake

        # Add the final turbine turbulence intensity field to the flow field object
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
        grid: FlowFieldGrid | FlowFieldPlanarGrid | PointsGrid,
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
            thrust_coefficient_i = self.turbine_thrust_coefficient(
                turbine_grid,
                turbine_grid_farm,
                turbine_grid_flow_field,
                i
            )
            axial_induction_i = self.turbine_axial_induction(
                turbine_grid,
                turbine_grid_farm,
                turbine_grid_flow_field,
                i
            )
            u_i = turbine_grid_flow_field.u_sorted[:, i:i+1]
            v_i = turbine_grid_flow_field.v_sorted[:, i:i+1]
            turbulence_intensity_i = \
                turbine_grid_flow_field.turbulence_intensity_field_sorted_avg[:, i:i+1]

            # Initialize the effective yaw angle
            self.effective_yaw_i = self.yaw_angle_i.copy()

            # Model calculations
            if self.enable_secondary_steering:
                added_yaw = wake_added_yaw(
                    u_i,
                    v_i,
                    turbine_grid_flow_field.u_initial_sorted,
                    turbine_grid.y_sorted[:, i:i+1] - self.y_i,
                    turbine_grid.z_sorted[:, i:i+1],
                    self.rotor_diameter_i,
                    self.hub_height_i,
                    thrust_coefficient_i,
                    self.TSR_i,
                    axial_induction_i,
                    flow_field.wind_shear,
                )
                self.effective_yaw_i += added_yaw

            deflection_field = self.deflection(
                turbulence_intensity_i,
                thrust_coefficient_i,
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
                    self.hub_height_i,
                    self.yaw_angle_i,
                    thrust_coefficient_i,
                    self.TSR_i,
                    axial_induction_i,
                    flow_field.wind_shear,
                )
            else:
                v_wake = np.zeros_like(flow_field.v_initial_sorted)
                w_wake = np.zeros_like(flow_field.w_initial_sorted)

            velocity_deficit = self.velocity_deficit(
                axial_induction_i,
                deflection_field,
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

            turbulence_intensity_field = self.turbulence(
                turbulence_intensity_field,
                grid.x_sorted,
                grid.y_sorted,
                axial_induction_i,
                np.where(velocity_deficit * flow_field.u_initial_sorted > 0.05, 1, 0),
            )

            flow_field.u_sorted = flow_field.u_initial_sorted - wake_field
            flow_field.v_sorted += v_wake
            flow_field.w_sorted += w_wake

        flow_field.turbulence_intensity_field_sorted = turbulence_intensity_field


# @profile
def rC(wind_veer, sigma_y, sigma_z, y, y_i, delta, z, HH, Ct, yaw, D):

    ## original
    # a = cosd(wind_veer) ** 2 / (2 * sigma_y ** 2) + sind(wind_veer) ** 2 / (2 * sigma_z ** 2)
    # b = -sind(2 * wind_veer) / (4 * sigma_y ** 2) + sind(2 * wind_veer) / (4 * sigma_z ** 2)
    # c = sind(wind_veer) ** 2 / (2 * sigma_y ** 2) + cosd(wind_veer) ** 2 / (2 * sigma_z ** 2)
    # r_squared = (
    #     a * (y - y_i - delta) ** 2
    #     - 2 * b * (y - y_i - delta) * (z - HH)
    #     + c * (z - HH) ** 2
    # )
    # C = 1 - np.sqrt(np.clip(1 - (Ct * cosd(yaw) / (8.0 * sigma_y * sigma_z / D ** 2)), 0.0, 1.0))

    ## Precalculate some parts
    # twox_sigmay_2 = 2 * sigma_y ** 2
    # twox_sigmaz_2 = 2 * sigma_z ** 2
    # a = cosd(wind_veer) ** 2 / (twox_sigmay_2) + sind(wind_veer) ** 2 / (twox_sigmaz_2)
    # b = -sind(2 * wind_veer) / (2 * twox_sigmay_2) + sind(2 * wind_veer) / (2 * twox_sigmaz_2)
    # c = sind(wind_veer) ** 2 / (twox_sigmay_2) + cosd(wind_veer) ** 2 / (twox_sigmaz_2)
    # delta_y = y - y_i - delta
    # delta_z = z - HH
    # r_squared = (a * (delta_y ** 2) - 2 * b * (delta_y) * (delta_z) + c * (delta_z ** 2))
    # C = 1 - np.sqrt(np.clip(1 - (Ct * cosd(yaw) / (8.0 * sigma_y * sigma_z / (D * D))), 0.0, 1.0))

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
    r_squared = ne.evaluate(
        "a * ((y - y_i - delta) ** 2) - 2 * b * (y - y_i - delta) * (z - HH) + c * ((z - HH) ** 2)"
    )
    d = np.clip(1 - (Ct * cosd(yaw) / ( 8.0 * sigma_y * sigma_z / (D * D) )), 0.0, 1.0)
    C = ne.evaluate("1 - sqrt(d)")
    return r_squared, C


def gaussian_function(C, r_squared, n, sigma):
    result = ne.evaluate("C * exp(-1 * r_squared ** n / (2 * sigma ** 2))")
    return result
