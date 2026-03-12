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
from floris.core.wake_model import BaseWakeModel
from floris.utilities import cosd, sind


NUM_EPS = fields(BaseModel).NUM_EPS.default

@define
class JensenJimenez(BaseWakeModel):

    # Jensen deficit model parameters
    we: float = field(default=0.05)

    # Jimenez deflection model parameters
    kd: float = field(default=0.05) # TODO: is this the same as we?
    ad: float = field(default=0.0)
    bd: float = field(default=0.0)

    # Crespo-Hernandez turbulence model parameters
    initial: float = field(converter=float, default=0.1)
    constant: float = field(converter=float, default=0.9)
    ai: float = field(converter=float, default=0.8)
    downstream: float = field(converter=float, default=-0.32)

    # Storage
    x_i: np.ndarray = field(init=False)
    y_i: np.ndarray = field(init=False)
    z_i: np.ndarray = field(init=False)

    yaw_angle_i: np.ndarray = field(init=False)
    hub_height_i: np.ndarray = field(init=False)
    rotor_diameter_i: np.ndarray = field(init=False)

    ambient_turbulence_intensities: np.ndarray = field(init=False)

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

        # u is 4-dimensional (n wind speeds, n turbines, grid res 1, grid res 2)
        # velocities is 3-dimensional (n turbines, grid res 1, grid res 2)

        # TODO: How much faster is numexpr? Is it worth it still worth it?

        x_i = self.x_i
        y_i = self.y_i
        z_i = self.z_i
        yaw_angle_i = self.yaw_angle_i
        hub_height_i = self.hub_height_i
        rotor_diameter_i = self.rotor_diameter_i # Must be unpacked for numexpr?

        rotor_radius = rotor_diameter_i / 2.0

        dx = ne.evaluate("x - x_i")
        dy = ne.evaluate("y - y_i - deflection_field_i")
        dz = ne.evaluate("z - z_i")

        we = self.we

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

    def deflection(
        self,
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        # TODO: Does it make more sense for x to simply be on the class? Seems to.
        # What should be passed in vs live on the class?
        """
        Calculates the deflection field of the wake in relation to the yaw of
        the turbine. This is coded as defined in [1].

        Args:
            x_locations (np.array): streamwise locations in wake
            y_locations (np.array): spanwise locations in wake
            z_locations (np.array): vertical locations in wake
                (not used in Jiménez)
            turbine (:py:class:`floris.core.turbine.Turbine`):
                Turbine object
            coord
                (:py:meth:`floris.core.turbine_map.TurbineMap.coords`):
                Spatial coordinates of wind turbine.
            flow_field
                (:py:class:`floris.core.flow_field.FlowField`):
                Flow field object.

        Returns:
            deflection (np.array): Deflected wake centerline.


        This function calculates the deflection of the entire flow field
        given the yaw angle and Ct of the current turbine
        """

        # Unpack for numexpr
        kd = self.kd
        ad = self.ad
        bd = self.bd

        x_i = self.x_i
        y_i = self.y_i
        yaw_i = self.yaw_angle_i
        rotor_diameter_i = self.rotor_diameter_i

        # angle of deflection
        xi_init = cosd(yaw_i) * sind(yaw_i) * ct_i / 2.0

        delta_x = ne.evaluate("x - x_i")
        A = ne.evaluate("15 * (2 * kd * delta_x / rotor_diameter_i + 1) ** 4.0 + xi_init ** 2.0")
        B = ne.evaluate("(30 * kd / rotor_diameter_i)")
        B = ne.evaluate("B * ( 2 * kd * delta_x / rotor_diameter_i + 1 ) ** 5.0")
        C = ne.evaluate("xi_init * rotor_diameter_i * (15 + xi_init ** 2.0)")
        D = ne.evaluate("30 * kd")

        yYaw_init = ne.evaluate("(xi_init * A / B) - (C / D)")
        deflection = ne.evaluate("yYaw_init + ad + bd * delta_x")

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

        # Calculate the velocity deficit sequentially from upstream to downstream turbines
        for i in range(grid.n_turbines):

            # Turbine quantities
            self.set_turbine_i(grid, farm, i)
            thrust_coefficient_i = self.turbine_thrust_coefficient(grid, farm, flow_field, i)
            axial_induction_i = self.turbine_axial_induction(grid, farm, flow_field, i)

            # Model calculations
            deflection_field = self.deflection(
                turbine_turbulence_intensity[:, i:i+1],
                thrust_coefficient_i,
                grid.x_sorted,
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
        flow_field_grid: FlowFieldGrid | FlowFieldPlanarGrid | PointsGrid,
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
        n_points = flow_field_grid.x_sorted.shape[1]
        ambient_turbulence_intensities = flow_field.turbulence_intensities[:, None, None, None]
        ambient_turbulence_intensities = np.repeat(ambient_turbulence_intensities, n_points, axis=1)
        turbulence_intensity_field = ambient_turbulence_intensities.copy()

        # Calculate the velocity deficit in the full grid sequentially from upstream to
        # downstream turbines
        for i in range(flow_field_grid.n_turbines):

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
            turbulence_intensity_i = \
                turbine_grid_flow_field.turbulence_intensity_field_sorted_avg[:, i:i+1]

            # Model calculations
            deflection_field = self.deflection(
                turbulence_intensity_i,
                thrust_coefficient_i,
                flow_field_grid.x_sorted,
            )

            velocity_deficit = self.velocity_deficit(
                axial_induction_i,
                deflection_field,
                turbulence_intensity_i,
                thrust_coefficient_i,
                flow_field_grid.x_sorted,
                flow_field_grid.y_sorted,
                flow_field_grid.z_sorted
            )

            wake_field = self.combination(
                wake_field,
                velocity_deficit * flow_field.u_initial_sorted
            )

            turbulence_intensity_field = self.turbulence(
                turbulence_intensity_field,
                flow_field_grid.x_sorted,
                flow_field_grid.y_sorted,
                axial_induction_i,
                np.where(velocity_deficit * flow_field.u_initial_sorted > 0.05, 1, 0),
            )

            flow_field.u_sorted = flow_field.u_initial_sorted - wake_field

        flow_field.turbulence_intensity_field_sorted = turbulence_intensity_field
