
from __future__ import annotations

import attrs
import matplotlib.path as mpltPath
import numpy as np
from attrs import define, field
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

from floris.core import (
    BaseClass,
    Grid,
)
from floris.type_dec import (
    floris_array_converter,
    NDArrayFloat,
)


@define
class FlowField(BaseClass):
    wind_speeds: NDArrayFloat = field(converter=floris_array_converter)
    wind_directions: NDArrayFloat = field(converter=floris_array_converter)
    wind_veer: float = field(converter=float)
    wind_shear: float = field(converter=float)
    air_density: float = field(converter=float)
    turbulence_intensities: NDArrayFloat = field(converter=floris_array_converter)
    reference_wind_height: float = field(converter=float)
    heterogeneous_inflow_config: dict = field(default=None)
    multidim_conditions: dict = field(default=None)

    n_findex: int = field(init=False)
    u_initial_sorted: NDArrayFloat = field(init=False, factory=lambda: np.array([]))
    v_initial_sorted: NDArrayFloat = field(init=False, factory=lambda: np.array([]))
    w_initial_sorted: NDArrayFloat = field(init=False, factory=lambda: np.array([]))
    u_sorted: NDArrayFloat = field(init=False, factory=lambda: np.array([]))
    v_sorted: NDArrayFloat = field(init=False, factory=lambda: np.array([]))
    w_sorted: NDArrayFloat = field(init=False, factory=lambda: np.array([]))
    u: NDArrayFloat = field(init=False, factory=lambda: np.array([]))
    v: NDArrayFloat = field(init=False, factory=lambda: np.array([]))
    w: NDArrayFloat = field(init=False, factory=lambda: np.array([]))
    het_map: list = field(init=False, default=None)
    dudz_initial_sorted: NDArrayFloat = field(init=False, factory=lambda: np.array([]))

    turbulence_intensity_field: NDArrayFloat = field(init=False, factory=lambda: np.array([]))
    turbulence_intensity_field_sorted: NDArrayFloat = field(
        init=False, factory=lambda: np.array([])
    )
    turbulence_intensity_field_sorted_avg: NDArrayFloat = field(
        init=False, factory=lambda: np.array([])
    )

    @turbulence_intensities.validator
    def turbulence_intensities_validator(
        self, instance: attrs.Attribute, value: NDArrayFloat
    ) -> None:

        # Check that the array is 1-dimensional
        if value.ndim != 1:
            raise ValueError(
                "turbulence_intensities must have 1-dimension"
            )

        # Check the turbulence intensity is length n_findex
        if len(value) != self.n_findex:
            raise ValueError("turbulence_intensities must be length n_findex")



    @wind_directions.validator
    def wind_directions_validator(self, instance: attrs.Attribute, value: NDArrayFloat) -> None:
        # Check that the array is 1-dimensional
        if self.wind_directions.ndim != 1:
            raise ValueError(
                "wind_directions must have 1-dimension"
            )

        """Using the validator method to keep the `n_findex` attribute up to date."""
        self.n_findex = value.size

    @wind_speeds.validator
    def wind_speeds_validator(self, instance: attrs.Attribute, value: NDArrayFloat) -> None:

        # Check that the array is 1-dimensional
        if self.wind_speeds.ndim != 1:
            raise ValueError(
                "wind_speeds must have 1-dimension"
            )

        """Confirm wind speeds and wind directions have the same length"""
        if len(self.wind_directions) != len(self.wind_speeds):
            raise ValueError(
                f"wind_directions (length = {len(self.wind_directions)}) and "
                f"wind_speeds (length = {len(self.wind_speeds)}) must have the same length"
            )

    @heterogeneous_inflow_config.validator
    def heterogeneous_config_validator(self, instance: attrs.Attribute, value: dict | None) -> None:
        """Using the validator method to check that the heterogeneous_inflow_config dictionary has
        the correct key-value pairs.
        """
        if value is None:
            return

        # Check that the correct keys are supplied for the heterogeneous_inflow_config dict
        for k in ["speed_multipliers", "x", "y"]:
            if k not in value.keys():
                raise ValueError(
                    "heterogeneous_inflow_config must contain entries for 'speed_multipliers',"
                    f"'x', and 'y', with 'z' optional. Missing '{k}'."
                )
        if "z" not in value:
            # If only a 2D case, add "None" for the z locations
            value["z"] = None

    @het_map.validator
    def het_map_validator(self, instance: attrs.Attribute, value: list | None) -> None:
        """Using this validator to make sure that the het_map has an interpolant defined for
        each findex.
        """
        if value is None:
            return

        if self.n_findex != np.array(value).shape[0]:
            raise ValueError(
                "The het_map's first dimension not equal to the FLORIS first dimension."
            )


    def __attrs_post_init__(self) -> None:
        if self.heterogeneous_inflow_config is not None:
            self.generate_heterogeneous_wind_map()


    def initialize_velocity_field(self, grid: Grid) -> None:

        # Create an initial wind profile as a function of height. The values here will
        # be multiplied with the wind speeds to give the initial wind field.
        # Since we use grid.z, this is a vertical plane for each turbine
        # Here, the profile is of shape (# turbines, N grid points, M grid points)
        # This velocity profile is 1.0 at the reference wind height and then follows wind
        # shear as an exponent.
        # NOTE: the convention of which dimension on the TurbineGrid is vertical and horizontal is
        # determined by this line. Since the right-most dimension on grid.z is storing the values
        # for height, using it here to apply the shear law makes that dimension store the vertical
        # wind profile.
        wind_profile_plane = (grid.z_sorted / self.reference_wind_height) ** self.wind_shear
        dwind_profile_plane = (
            self.wind_shear
            * (1 / self.reference_wind_height) ** self.wind_shear
            * np.power(
                grid.z_sorted,
                (self.wind_shear - 1),
                where=grid.z_sorted != 0.0
            )
        )
        # If no heterogeneous inflow defined, then set all speeds ups to 1.0
        if self.het_map is None:
            speed_ups = 1.0

        # If heterogeneous flow data is given, the speed ups at the defined
        # grid locations are determined in either 2 or 3 dimensions.
        else:
            bounds = np.array(list(zip(
                self.heterogeneous_inflow_config['x'],
                self.heterogeneous_inflow_config['y']
            )))
            hull = ConvexHull(bounds)
            polygon = Polygon(bounds[hull.vertices])
            path = mpltPath.Path(polygon.boundary.coords)
            points = np.column_stack(
                (
                    grid.x_sorted_inertial_frame.flatten(),
                    grid.y_sorted_inertial_frame.flatten(),
                )
            )
            inside = path.contains_points(points)
            if not np.all(inside):
                self.logger.warning(
                    "The calculated flow field contains points outside of the the user-defined "
                    "heterogeneous inflow bounds. For these points, the interpolated value has "
                    "been filled with the freestream wind speed. If this is not the desired "
                    "behavior, the user will need to expand the heterogeneous inflow bounds to "
                    "fully cover the calculated flow field area."
                )

            if len(self.het_map[0].points[0]) == 2:
                speed_ups = self.calculate_speed_ups(
                    self.het_map,
                    grid.x_sorted_inertial_frame,
                    grid.y_sorted_inertial_frame
                )
            elif len(self.het_map[0].points[0]) == 3:
                speed_ups = self.calculate_speed_ups(
                    self.het_map,
                    grid.x_sorted_inertial_frame,
                    grid.y_sorted_inertial_frame,
                    grid.z_sorted
                )

        # Create the sheer-law wind profile
        # This array is of shape (# wind directions, # wind speeds, grid.template_array)
        # Since generally grid.template_array may be many different shapes, we use transposes
        # here to do broadcasting from left to right (transposed), and then transpose back.
        # The result is an array the wind speed and wind direction dimensions on the left side
        # of the shape and the grid.template array on the right
        self.u_initial_sorted = (self.wind_speeds.T * wind_profile_plane.T).T * speed_ups
        self.dudz_initial_sorted = (self.wind_speeds.T * dwind_profile_plane.T).T * speed_ups

        self.v_initial_sorted = np.zeros(
            np.shape(self.u_initial_sorted),
            dtype=self.u_initial_sorted.dtype
        )
        self.w_initial_sorted = np.zeros(
            np.shape(self.u_initial_sorted),
            dtype=self.u_initial_sorted.dtype
        )

        self.u_sorted = self.u_initial_sorted.copy()
        self.v_sorted = self.v_initial_sorted.copy()
        self.w_sorted = self.w_initial_sorted.copy()

        self.turbulence_intensity_field = self.turbulence_intensities[:, None, None, None]
        self.turbulence_intensity_field = np.repeat(
            self.turbulence_intensity_field,
            grid.n_turbines,
            axis=1
        )

        self.turbulence_intensity_field_sorted = self.turbulence_intensity_field.copy()

    def finalize(self, unsorted_indices):
        self.u = np.take_along_axis(self.u_sorted, unsorted_indices, axis=1)
        self.v = np.take_along_axis(self.v_sorted, unsorted_indices, axis=1)
        self.w = np.take_along_axis(self.w_sorted, unsorted_indices, axis=1)

        self.turbulence_intensity_field = np.mean(
            np.take_along_axis(
                self.turbulence_intensity_field_sorted,
                unsorted_indices,
                axis=1
            ),
            axis=(2,3)
        )

    def calculate_speed_ups(self, het_map, x, y, z=None):
        if z is not None:
            # Calculate the 3-dimensional speed ups; squeeze is needed as the generator
            # adds an extra dimension
            speed_ups = np.squeeze(
                [het_map[i](x[i:i+1], y[i:i+1], z[i:i+1]) for i in range( len(het_map))],
                axis=1,
            )

        else:
            # Calculate the 2-dimensional speed ups; squeeze is needed as the generator
            # adds an extra dimension
            speed_ups = np.squeeze(
                [het_map[i](x[i:i+1], y[i:i+1]) for i in range(len(het_map))],
                axis=1,
            )

        return speed_ups

    def generate_heterogeneous_wind_map(self):
        """This function creates the heterogeneous interpolant used to calculate heterogeneous
        inflows. The interpolant is for computing wind speed based on an x and y location in the
        flow field. This is computed using SciPy's LinearNDInterpolator and uses a fill value
        equal to the freestream for interpolated values outside of the user-defined heterogeneous
        map bounds.

        Args:
            heterogeneous_inflow_config (dict): The heterogeneous inflow configuration dictionary.
            The configuration should have the following inputs specified.
                - **speed_multipliers** (list): A list of speed up factors that will multiply
                    the specified freestream wind speed. This 2-dimensional array should have an
                    array of multiplicative factors defined for each wind direction.
                - **x** (list): A list of x locations at which the speed up factors are defined.
                - **y**: A list of y locations at which the speed up factors are defined.
                - **z** (optional): A list of z locations at which the speed up factors are defined.
        """
        speed_multipliers = self.heterogeneous_inflow_config['speed_multipliers']
        x = self.heterogeneous_inflow_config['x']
        y = self.heterogeneous_inflow_config['y']
        z = self.heterogeneous_inflow_config['z']

        if z is not None:
            # Compute the 3-dimensional interpolants for each wind direction
            # Linear interpolation is used for points within the user-defined area of values,
            # while the freestream wind speed is used for points outside that region
            in_region = [
                LinearNDInterpolator(list(zip(x, y, z)), multiplier, fill_value=1.0)
                for multiplier in speed_multipliers
            ]
        else:
            # Compute the 2-dimensional interpolants for each wind direction
            # Linear interpolation is used for points within the user-defined area of values,
            # while the freestream wind speed is used for points outside that region
            in_region = [
                LinearNDInterpolator(list(zip(x, y)), multiplier, fill_value=1.0)
                for multiplier in speed_multipliers
            ]

        self.het_map = in_region
