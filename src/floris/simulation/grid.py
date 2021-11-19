from __future__ import annotations

from abc import ABC, abstractmethod

import attr
import numpy as np
import numpy.typing as npt
from numpy import newaxis as na

from floris.utilities import Vec3, cosd, sind, attrs_array_converter


NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt = npt.NDArray[np.int_]


@attr.s(auto_attribs=True)
class Grid(ABC):
    """
    Grid should establish domain bounds based on given criteria,
    and develop three arrays to contain components of the grid
    locations in space.

    This could be generalized to any number of dimensions to be
    used by perhaps a turbulence field.

    The grid will have to be reestablished for each wind direction since the planform
    area of the farm will be different.

    Args:
        turbine_coordinates (`list[Vec3]`): The collection of turbine coordinate (`Vec3`) objects.
        reference_turbine_diameter (:py:obj:`float`): The reference turbine's rotor diameter.
        grid_resolution (:py:obj:`int`): The number of points on each turbine
    """

    # TODO: We'll want to consider how this expands to multiple turbine types
    turbine_coordinates: list[Vec3] = attr.ib(on_setattr=attr.setters.validate)
    reference_turbine_diameter: float
    grid_resolution: int
    wind_directions: NDArrayFloat = attr.ib(
        converter=attrs_array_converter, on_setattr=(attr.setters.convert, attr.setters.validate)
    )
    wind_speeds: NDArrayFloat = attr.ib(
        converter=attrs_array_converter, on_setattr=(attr.setters.convert, attr.setters.validate)
    )

    n_turbines: int = attr.ib(init=False)
    n_wind_speeds: int = attr.ib(init=False)
    n_wind_directions: int = attr.ib(init=False)
    turbine_coordinates_array: NDArrayFloat = attr.ib(init=False)
    x: NDArrayFloat = attr.ib(init=False)
    y: NDArrayFloat = attr.ib(init=False)
    z: NDArrayFloat = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        self.turbine_coordinates_array = np.array([c.elements for c in self.turbine_coordinates])

    @turbine_coordinates.validator
    def check_coordinates(self, instance: attr.Attribute, value: list[Vec3]) -> None:
        """Ensures all elements are `Vec3` objects and keeps the `n_turbines` attribute up to date."""
        types = np.unique([isinstance(c, Vec3) for c in value])
        if not all(types):
            raise TypeError("'turbine_coordinates' should be all `Vec3` objects!")

        self.n_turbines = len(value)

    @wind_speeds.validator
    def wind_speeds_validator(self, instance: attr.Attribute, value: NDArrayFloat) -> None:
        """Using the validator method to keep the `n_wind_speeds` attribute up to date."""
        self.n_wind_speeds = value.size

    @wind_directions.validator
    def wind_directionss_validator(self, instance: attr.Attribute, value: NDArrayFloat) -> None:
        """Using the validator method to keep the `n_wind_directions` attribute up to date."""
        self.n_wind_directions = value.size

    # x are the locations in space in the primary direction (typically the direction of the wind)
    # y are the locations in space in the lateral direction
    # z are the locations in space in the vertical direction
    # u are the velocity components at each point in space
    # v are the velocity components at each point in space
    # w are the velocity components at each point in space
    # all of these arrays are the same size

    # @abstractmethod
    # def set_bounds(self) -> None:
    #     # TODO: Should this be called "compute_bounds?"
    #     #   anything set_ could require an argument to set a value
    #     #   other functions that set variables based on previous inputs could be "compute_"
    #     #   anything that returns values, even if they are computed on the fly, could be get_ (instead of @property)
    #     raise NotImplementedError("Grid.set_bounds")

    # def get_bounds(self) -> List[float]:
    #     """
    #     The minimum and maximum values of the bounds of the computational domain.
    #     """
    #     return [self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax]

    @abstractmethod
    def set_grid(self) -> None:
        raise NotImplementedError("Grid.set_grid")

    def rotate_fields(self, wd: int | float) -> None:
        # Find center of rotation
        x_center_of_rotation = (np.min(self.x) + np.max(self.x)) / 2
        y_center_of_rotation = (np.min(self.y) + np.max(self.y)) / 2

        angle = ((wd - 270) % 360 + 360) % 360
        # angle = (wd - 270) % 360 # Is this the same as above?

        # Rotate grid points
        x_offset = self.x - x_center_of_rotation
        y_offset = self.y - y_center_of_rotation
        mesh_x_rotated = x_offset * cosd(angle) - y_offset * sind(angle) + x_center_of_rotation
        mesh_y_rotated = x_offset * sind(angle) + y_offset * cosd(angle) + y_center_of_rotation

        # print(np.shape(mesh_x_rotated))
        # lkj
        self.x = mesh_x_rotated
        self.y = mesh_y_rotated

    @staticmethod
    def rotate_turbine_locations(
        coords: NDArrayFloat | list[Vec3], wd: int | float
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Rotates the turbine locations with respect to a wind direction.

        Args:
            coords (NDArrayFloat | list[Vec3]): Either the `Grid.turbine_coordinates`
            `list` of `Vec3` objects or the `Grid.turbine_coordinates_array` 2D array object.
            wd (int): The wind direction to rotate the coordinate field.

        Returns:
            tuple[NDArrayFloat, NDArrayFloat]: The rotated x and y coordinates
        """
        if isinstance(coords, NDArrayFloat):
            x_coord, y_coord, _ = coords.T
        else:
            x_coord = np.array([c.x1 for c in coords])
            y_coord = np.array([c.x2 for c in coords])

        # Find center of rotation
        x_center_of_rotation = (np.min(x_coord) + np.max(x_coord)) / 2
        y_center_of_rotation = (np.min(y_coord) + np.max(y_coord)) / 2

        angle = ((wd - 270) % 360 + 360) % 360
        # angle = (wd - 270) % 360 # Is this the same as above?

        # Rotate turbine coordinates
        x_coord_offset = x_coord - x_center_of_rotation
        y_coord_offset = y_coord - y_center_of_rotation
        x_coord_rotated = x_coord_offset * cosd(angle) - y_coord_offset * sind(angle) + x_center_of_rotation
        y_coord_rotated = x_coord_offset * sind(angle) + y_coord_offset * cosd(angle) + y_center_of_rotation
        return x_coord_rotated, y_coord_rotated


@attr.s(auto_attribs=True)
class TurbineGrid(Grid):
    """See `Grid` for more details.

    Args:
        turbine_coordinates (`list[Vec3]`): The collection of turbine coordinate (`Vec3`) objects.
        reference_turbine_diameter (:py:obj:`float`): The reference turbine's rotor diameter.
        wind_directions (`list[float]`): The input wind directions
        wind_speeds (`list[float]`): The input wind speeds
        grid_resolution (:py:obj:`int`): The number of points on each turbine
    """

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        self.set_grid()

    def set_grid(self) -> None:
        """
        Create grid points at each turbine for each wind direction and wind speed in the simulation.
        This creates the underlying data structure for the calculation.

        arrays have shape (n wind directions, n wind speeds, n turbines, m grid spanwise, m grid vertically)
        - dimension 1: each wind direction
        - dimension 2: each wind speed
        - dimension 3: each turbine
        - dimension 4: number of points in the spanwise direction (ngrid)
        - dimension 5: number of points in the vertical dimension (ngrid)

        For example
        - x is [n wind direction, n wind speeds, n turbines, x-component of the points in the spanwise direction, x-component of the points in the vertical direction]
        - y is [n wind direction, n wind speeds, n turbines, y-component of the points in the spanwise direction, y-component of the points in the vertical direction]

        The x,y,z arrays contain the actual locations in that direction.

        # -   **self.grid_resolution** (*int*, optional): The square root of the number
        #             of points to use on the turbine grid. This number will be
        #             squared so that the points can be evenly distributed.
        #             Defaults to 5.

        If the grid conforms to the sequential solver interface, it must be sorted from upstream to downstream
        """
        # TODO: Where should we locate the coordinate system? Currently, its at
        # the foot of the turbine where the tower meets the ground.

        # Calculate the difference in given wind direction from 270 / West
        wind_deviation_from_west = -1 * ((self.wind_directions - 270) % 360 + 360) % 360
        wind_deviation_from_west = np.reshape(wind_deviation_from_west, (self.n_wind_directions, 1, 1))

        # Construct the arrays storing the turbine locations
        x_coordinates, y_coordinates, z_coordinates = self.turbine_coordinates_array.T
        x_coordinates = x_coordinates[None, None, :]
        y_coordinates = y_coordinates[None, None, :]
        z_coordinates = z_coordinates[None, None, :]

        # Find center of rotation - this is the center of box bounding all of the turbines
        x_center_of_rotation = (np.min(x_coordinates) + np.max(x_coordinates)) / 2
        y_center_of_rotation = (np.min(y_coordinates) + np.max(y_coordinates)) / 2

        # Rotate turbine coordinates about the center
        x_coord_offset = x_coordinates - x_center_of_rotation
        y_coord_offset = y_coordinates - y_center_of_rotation
        x_coord_rotated = (
            x_coord_offset * cosd(wind_deviation_from_west)
            - y_coord_offset * sind(wind_deviation_from_west)
            + x_center_of_rotation
        )
        y_coord_rotated = (
            x_coord_offset * sind(wind_deviation_from_west)
            + y_coord_offset * cosd(wind_deviation_from_west)
            + y_center_of_rotation
        )

        # -   **rloc** (*float, optional): A value, from 0 to 1, that determines
        #         the width/height of the grid of points on the rotor as a ratio of
        #         the rotor radius.
        #         Defaults to 0.5.

        # Create the data for the turbine grids
        radius_ratio = 0.5
        disc_area_radius = radius_ratio * self.reference_turbine_diameter / 2
        template_grid = np.ones(
            (
                self.n_wind_directions,
                self.n_wind_speeds,
                self.n_turbines,
                self.grid_resolution,
                self.grid_resolution,
            )
        )

        # Calculate the radial distance from the center of the turbine rotor
        disc_grid = np.linspace(-1 * disc_area_radius, disc_area_radius, self.grid_resolution)
        template_rotor = template_grid * (disc_grid * np.ones((self.grid_resolution, self.grid_resolution)))

        # Construct the turbine grids
        # Here, they are already rotated to the correct orientation for each wind direction
        _x = x_coord_rotated[:, :, :, None, None] * template_grid
        _y = y_coord_rotated[:, :, :, None, None] + template_rotor
        _z = z_coordinates[:, :, :, None, None] + template_rotor

        # Sort the turbines at each wind direction

        # Get the sorted indices for the x coordinates. These are the indices
        # to sort the turbines from upstream to downstream for all wind directions.
        # Also, store the indices to sort them back for when the calculation finishes.
        self.sorted_indices = _x.argsort(axis=2)
        self.unsorted_indices = self.sorted_indices.argsort(axis=2)

        # Put the turbines into the final arrays in their sorted order
        self.x = np.take_along_axis(_x, self.sorted_indices, axis=2)
        self.y = np.take_along_axis(_y, self.sorted_indices, axis=2)
        self.z = np.take_along_axis(_z, self.sorted_indices, axis=2)

    def finalize(self):
        # Replace the turbines in their unsorted order so that
        # we can report values in a sane way.
        self.x = np.take_along_axis(self.x, self.unsorted_indices, axis=2)
        self.y = np.take_along_axis(self.y, self.unsorted_indices, axis=2)
        self.z = np.take_along_axis(self.z, self.unsorted_indices, axis=2)


# class FlowFieldGrid(Grid):
#     """
#     Primarily used by the Curl model and for visualization

#     Args:
#         grid_resolution (`Vec3`): The number of grid points to be created in each direction.
#         turbine_coordinates (`list[Vec3]`): The collection of turbine coordinate (`Vec3`) objects.
#         reference_turbine_diameter (:py:obj:`float`): The reference turbine's rotor diameter.
#         grid_resolution (:py:obj:`int`): The number of points on each turbine
#     """

#     grid_resolution: Vec3
#     xmin: float = attr.ib(init=False)
#     xmax: float = attr.ib(init=False)
#     ymin: float = attr.ib(init=False)
#     ymax: float = attr.ib(init=False)
#     zmin: float = attr.ib(init=False)
#     zmax: float = attr.ib(init=False)

#     def __attrs_post_init__(self) -> None:
#         super().__attrs_post_init__()
#         self.set_bounds()
#         self.set_grid()

#     def set_bounds(self) -> None:
#         # TODO: Should this be called "compute_bounds?"
#         #   anything set_ could require an argument to set a value
#         #   other functions that set variables based on previous inputs could be "compute_"
#         #   anything that returns values, even if they are computed on the fly, could be get_ (instead of @property)
#         """
#         Calculates the domain bounds for the current wake model. The bounds
#         are calculated based on preset extents from the
#         given layout. The bounds consist of the minimum and maximum values
#         in the x-, y-, and z-directions.

#         If the Curl model is used, the predefined bounds are always set.
#         """
#         # For the curl model, bounds are hard coded
#         eps = 0.1
#         self.xmin = min(self.turbine_coordinates_array[:, 0]) - 2 * self.reference_turbine_diameter
#         self.xmax = max(self.turbine_coordinates_array[:, 0]) + 10 * self.reference_turbine_diameter
#         self.ymin = min(self.turbine_coordinates_array[:, 1]) - 2 * self.reference_turbine_diameter
#         self.ymax = max(self.turbine_coordinates_array[:, 1]) + 2 * self.reference_turbine_diameter
#         self.zmin = 0 + eps
#         self.zmax = 6 * self.reference_wind_height

#     def set_grid(self) -> None:
#         """
#         Create a structured grid for the entire flow field domain.
#         resolution: Vec3
#         """
#         x_points = np.linspace(self.xmin, self.xmax, int(self.grid_resolution.x1))
#         y_points = np.linspace(self.ymin, self.ymax, int(self.grid_resolution.x2))
#         z_points = np.linspace(self.zmin, self.zmax, int(self.grid_resolution.x3))
#         self.x, self.y, self.z = np.meshgrid(x_points, y_points, z_points, indexing="ij")
