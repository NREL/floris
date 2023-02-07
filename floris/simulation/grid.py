# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import attrs
import numpy as np
from attrs import define, field

from floris.type_dec import (
    floris_array_converter,
    floris_float_type,
    NDArrayFloat,
    NDArrayInt,
)
from floris.utilities import rotate_coordinates_rel_west, Vec3


@define
class Grid(ABC):
    """
    Grid should establish domain bounds based on given criteria,
    and develop three arrays to contain components of the grid
    locations in space.

    This could be generalized to any number of dimensions to be
    used by perhaps a turbulence field.

    The grid will have to be reestablished for each wind direction since the planform
    area of the farm will be different.

    x are the locations in space in the primary direction (typically the direction of the wind)
    y are the locations in space in the lateral direction
    z are the locations in space in the vertical direction
    u are the velocity components at each point in space
    v are the velocity components at each point in space
    w are the velocity components at each point in space
    all of these arrays are the same size

    Args:
        turbine_coordinates (`list[Vec3]`): The collection of turbine coordinate (`Vec3`) objects.
        reference_turbine_diameter (:py:obj:`float`): The reference turbine's rotor diameter.
        grid_resolution (:py:obj:`int` | :py:obj:`Iterable(int,)`): Grid resolution specific
            to each grid type.
        wind_directions (:py:obj:`NDArrayFloat`): Wind directions supplied by the user.
        wind_speeds (:py:obj:`NDArrayFloat`): Wind speeds supplied by the user.
        time_series (:py:obj:`bool`): True/false flag to indicate whether the supplied wind
            data is a time series.
    """
    turbine_coordinates: list[Vec3] = field()
    reference_turbine_diameter: float
    grid_resolution: int | Iterable = field()
    wind_directions: NDArrayFloat = field(converter=floris_array_converter)
    wind_speeds: NDArrayFloat = field(converter=floris_array_converter)
    time_series: bool = field()

    n_turbines: int = field(init=False)
    n_wind_speeds: int = field(init=False)
    n_wind_directions: int = field(init=False)
    turbine_coordinates_array: NDArrayFloat = field(init=False)
    x: NDArrayFloat = field(init=False, default=[])
    y: NDArrayFloat = field(init=False, default=[])
    z: NDArrayFloat = field(init=False, default=[])
    x_sorted: NDArrayFloat = field(init=False)
    y_sorted: NDArrayFloat = field(init=False)
    z_sorted: NDArrayFloat = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.turbine_coordinates_array = np.array([c.elements for c in self.turbine_coordinates])

    @turbine_coordinates.validator
    def check_coordinates(self, instance: attrs.Attribute, value: list[Vec3]) -> None:
        """
        Ensures all elements are `Vec3` objects and keeps the `n_turbines`
        attribute up to date.
        """
        types = np.unique([isinstance(c, Vec3) for c in value])
        if not all(types):
            raise TypeError("'turbine_coordinates' must be `Vec3` objects.")

        self.n_turbines = len(value)

    @wind_speeds.validator
    def wind_speeds_validator(self, instance: attrs.Attribute, value: NDArrayFloat) -> None:
        """Using the validator method to keep the `n_wind_speeds` attribute up to date."""
        if self.time_series:
            self.n_wind_speeds = 1
        else:
            self.n_wind_speeds = value.size

    @wind_directions.validator
    def wind_directions_validator(self, instance: attrs.Attribute, value: NDArrayFloat) -> None:
        """Using the validator method to keep the `n_wind_directions` attribute up to date."""
        self.n_wind_directions = value.size

    @grid_resolution.validator
    def grid_resolution_validator(self, instance: attrs.Attribute, value: int | Iterable) -> None:
        # TODO move this to the grid types and off of the base class
        """Check that grid resolution is given as int or Vec3 with int components."""
        if isinstance(value, int) and type(self) is TurbineGrid:
            return
        elif isinstance(value, Iterable) and type(self) is FlowFieldPlanarGrid:
            assert type(value[0]) is int
            assert type(value[1]) is int
        elif isinstance(value, Iterable) and type(self) is FlowFieldGrid:
            assert type(value[0]) is int
            assert type(value[1]) is int
            assert type(value[2]) is int
        else:
            raise TypeError("`grid_resolution` must be of type int or Iterable(int,)")

    @abstractmethod
    def set_grid(self) -> None:
        raise NotImplementedError("Grid.set_grid")

@define
class TurbineGrid(Grid):
    """See `Grid` for more details.

    Args:
        turbine_coordinates (`list[Vec3]`): The collection of turbine coordinate (`Vec3`) objects.
        reference_turbine_diameter (:py:obj:`float`): The reference turbine's rotor diameter.
        wind_directions (`list[float]`): The input wind directions
        wind_speeds (`list[float]`): The input wind speeds
        grid_resolution (:py:obj:`int`): The number of points on each turbine
    """
    # TODO: describe these and the differences between `sorted_indices` and `sorted_coord_indices`
    sorted_indices: NDArrayInt = field(init=False)
    sorted_coord_indices: NDArrayInt = field(init=False)
    unsorted_indices: NDArrayInt = field(init=False)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        self.set_grid()

    def set_grid(self) -> None:
        """
        Create grid points at each turbine for each wind direction and wind speed in the simulation.
        This creates the underlying data structure for the calculation.

        arrays have shape
        (n wind directions, n wind speeds, n turbines, m grid spanwise, m grid vertically)
        - dimension 1: each wind direction
        - dimension 2: each wind speed
        - dimension 3: each turbine
        - dimension 4: number of points in the spanwise direction (ngrid)
        - dimension 5: number of points in the vertical dimension (ngrid)

        For example
        - x is [
            n wind direction,
            n wind speeds,
            n turbines,
            x-component of the points in the spanwise direction,
            x-component of the points in the vertical direction
        ]
        - y is [
            n wind direction,
            n wind speeds,
            n turbines,
            y-component of the points in the spanwise direction,
            y-component of the points in the vertical direction
        ]

        The x,y,z arrays contain the actual locations in that direction.

        # -   **self.grid_resolution** (*int*, optional): The square root of the number
        #             of points to use on the turbine grid. This number will be
        #             squared so that the points can be evenly distributed.
        #             Defaults to 5.

        If the grid conforms to the sequential solver interface,
        it must be sorted from upstream to downstream

        In a y-z plane on the rotor swept area, the -2 dimension is a column of
        points and the -1 dimension is the row number.
        So the following line prints the 0'th column of the the 0'th turbine's grid:
        print(grid.y_sorted[0,0,0,0,:])
        print(grid.z_sorted[0,0,0,0,:])
        And this line prints a single point
        print(grid.y_sorted[0,0,0,0,0])
        print(grid.z_sorted[0,0,0,0,0])
        Note that the x coordinates are all the same for the rotor plane.

        """
        # TODO: Where should we locate the coordinate system? Currently, its at
        # the foot of the turbine where the tower meets the ground.

        # These are the rotated coordinates of the wind turbines based on the wind direction
        x, y, z = rotate_coordinates_rel_west(self.wind_directions, self.turbine_coordinates_array)

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
            ),
            dtype=floris_float_type
        )
        # Calculate the radial distance from the center of the turbine rotor.
        # If a grid resolution of 1 is selected, create a disc_grid of zeros, as
        # np.linspace would just return the starting value of -1 * disc_area_radius
        # which would place the point below the center of the rotor.
        if self.grid_resolution == 1:
            disc_grid = np.zeros((np.shape(disc_area_radius)[0], 1 ))
        else:
            disc_grid = np.linspace(
                -1 * disc_area_radius,
                disc_area_radius,
                self.grid_resolution,
                dtype=floris_float_type,
                axis=1
            )
        # Construct the turbine grids
        # Here, they are already rotated to the correct orientation for each wind direction
        _x = x[:, :, :, None, None] * template_grid

        ones_grid = np.ones(
            (self.n_turbines, self.grid_resolution, self.grid_resolution),
            dtype=floris_float_type
        )
        _y = y[:, :, :, None, None] + template_grid * ( disc_grid[None, None, :, :, None])
        _z = z[:, :, :, None, None] + template_grid * ( disc_grid[:, None, :] * ones_grid )

        # Sort the turbines at each wind direction

        # Get the sorted indices for the x coordinates. These are the indices
        # to sort the turbines from upstream to downstream for all wind directions.
        # Also, store the indices to sort them back for when the calculation finishes.
        self.sorted_indices = _x.argsort(axis=2)
        self.sorted_coord_indices = x.argsort(axis=2)
        self.unsorted_indices = self.sorted_indices.argsort(axis=2)

        # Put the turbines into the final arrays in their sorted order
        self.x_sorted = np.take_along_axis(_x, self.sorted_indices, axis=2)
        self.y_sorted = np.take_along_axis(_y, self.sorted_indices, axis=2)
        self.z_sorted = np.take_along_axis(_z, self.sorted_indices, axis=2)

        self.x = np.take_along_axis(self.x_sorted, self.unsorted_indices, axis=2)
        self.y = np.take_along_axis(self.y_sorted, self.unsorted_indices, axis=2)
        self.z = np.take_along_axis(self.z_sorted, self.unsorted_indices, axis=2)

@define
class FlowFieldGrid(Grid):
    """
    Args:
        grid_resolution (`Vec3`): The number of grid points to be created in each direction.
        turbine_coordinates (`list[Vec3]`): The collection of turbine coordinate (`Vec3`) objects.
        reference_turbine_diameter (:py:obj:`float`): The reference turbine's rotor diameter.
        grid_resolution (:py:obj:`int`): The number of points on each turbine
    """

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        self.set_grid()

    def set_grid(self) -> None:
        """
        Create a structured grid for the entire flow field domain.
        resolution: Vec3

        Calculates the domain bounds for the current wake model. The bounds
        are calculated based on preset extents from the
        given layout. The bounds consist of the minimum and maximum values
        in the x-, y-, and z-directions.

        If the Curl model is used, the predefined bounds are always set.

        First, sort the turbines so that we know the bounds in the correct orientation.
        Then, create the grid based on this wind-from-left orientation
        """

        # These are the rotated coordinates of the wind turbines based on the wind direction
        x, y, z = rotate_coordinates_rel_west(self.wind_directions, self.turbine_coordinates_array)

        # Construct the arrays storing the grid points
        eps = 0.01
        xmin = min(x[0,0]) - 2 * self.reference_turbine_diameter
        xmax = max(x[0,0]) + 10 * self.reference_turbine_diameter
        ymin = min(y[0,0]) - 2 * self.reference_turbine_diameter
        ymax = max(y[0,0]) + 2 * self.reference_turbine_diameter
        zmin = 0 + eps
        zmax = 6 * max(z[0,0])

        x_points, y_points, z_points = np.meshgrid(
            np.linspace(xmin, xmax, int(self.grid_resolution[0])),
            np.linspace(ymin, ymax, int(self.grid_resolution[1])),
            np.linspace(zmin, zmax, int(self.grid_resolution[2])),
            indexing="ij"
        )

        self.x_sorted = x_points[None, None, :, :, :]
        self.y_sorted = y_points[None, None, :, :, :]
        self.z_sorted = z_points[None, None, :, :, :]

@define
class FlowFieldPlanarGrid(Grid):
    """
    Args:
        grid_resolution (`Vec3`): The number of grid points to be created in each direction.
        turbine_coordinates (`list[Vec3]`): The collection of turbine coordinate (`Vec3`) objects.
        reference_turbine_diameter (:py:obj:`float`): The reference turbine's rotor diameter.
        grid_resolution (:py:obj:`int`): The number of points on each turbine
    """
    normal_vector: str = field()
    planar_coordinate: float = field()
    x1_bounds: tuple = field(default=None)
    x2_bounds: tuple = field(default=None)

    sorted_indices: NDArrayInt = field(init=False)
    unsorted_indices: NDArrayInt = field(init=False)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        self.set_grid()

    def set_grid(self) -> None:
        """
        Create a structured grid for the entire flow field domain.
        resolution: Vec3

        Calculates the domain bounds for the current wake model. The bounds
        are calculated based on preset extents from the
        given layout. The bounds consist of the minimum and maximum values
        in the x-, y-, and z-directions.

        If the Curl model is used, the predefined bounds are always set.

        First, sort the turbines so that we know the bounds in the correct orientation.
        Then, create the grid based on this wind-from-left orientation
        """
        # These are the rotated coordinates of the wind turbines based on the wind direction
        x, y, z = rotate_coordinates_rel_west(self.wind_directions, self.turbine_coordinates_array)

        max_diameter = np.max(self.reference_turbine_diameter)

        if self.normal_vector == "z":  # Rules of thumb for horizontal plane
            if self.x1_bounds is None:
                self.x1_bounds = (np.min(x) - 2 * max_diameter, np.max(x) + 10 * max_diameter)

            if self.x2_bounds is None:
                self.x2_bounds = (np.min(y) - 2 * max_diameter, np.max(y) + 2 * max_diameter)

            # TODO figure out proper z spacing for GCH, currently set to +/- 10.0
            x_points, y_points, z_points = np.meshgrid(
                np.linspace(self.x1_bounds[0], self.x1_bounds[1], int(self.grid_resolution[0])),
                np.linspace(self.x2_bounds[0], self.x2_bounds[1], int(self.grid_resolution[1])),
                np.array([
                    float(self.planar_coordinate) - 10.0,
                    float(self.planar_coordinate),
                    float(self.planar_coordinate) + 10.0
                ]),
                indexing="ij"
            )

            self.x_sorted = x_points[None, None, :, :, :]
            self.y_sorted = y_points[None, None, :, :, :]
            self.z_sorted = z_points[None, None, :, :, :]

        elif self.normal_vector == "x":  # Rules of thumb for cross plane
            if self.x1_bounds is None:
                self.x1_bounds = (np.min(y) - 2 * max_diameter, np.max(y) + 2 * max_diameter)

            if self.x2_bounds is None:
                self.x2_bounds = (0.001, 6 * np.max(z))

            x_points, y_points, z_points = np.meshgrid(
                np.array([float(self.planar_coordinate)]),
                np.linspace(self.x1_bounds[0], self.x1_bounds[1], int(self.grid_resolution[0])),
                np.linspace(self.x2_bounds[0], self.x2_bounds[1], int(self.grid_resolution[1])),
                indexing="ij"
            )

            self.x_sorted = x_points[None, None, :, :, :]
            self.y_sorted = y_points[None, None, :, :, :]
            self.z_sorted = z_points[None, None, :, :, :]

        elif self.normal_vector == "y":  # Rules of thumb for y plane
            if self.x1_bounds is None:
                self.x1_bounds = (np.min(x) - 2 * max_diameter, np.max(x) + 10 * max_diameter)

            if self.x2_bounds is None:
                self.x2_bounds = (0.001, 6 * np.max(z))

            x_points, y_points, z_points = np.meshgrid(
                np.linspace(self.x1_bounds[0], self.x1_bounds[1], int(self.grid_resolution[0])),
                np.array([float(self.planar_coordinate)]),
                np.linspace(self.x2_bounds[0], self.x2_bounds[1], int(self.grid_resolution[1])),
                indexing="ij"
            )

            self.x_sorted = x_points[None, None, :, :, :]
            self.y_sorted = y_points[None, None, :, :, :]
            self.z_sorted = z_points[None, None, :, :, :]

        # self.sorted_indices = self.x.argsort(axis=2)
        # self.unsorted_indices = self.sorted_indices.argsort(axis=2)

        # Put the turbines into the final arrays in their sorted order
        # self.x = np.take_along_axis(self.x, self.sorted_indices, axis=2)
        # self.y = np.take_along_axis(self.y, self.sorted_indices, axis=2)
        # self.z = np.take_along_axis(self.z, self.sorted_indices, axis=2)

    # def finalize(self):
        # sorted_indices = self.x.argsort(axis=2)
        # unsorted_indices = sorted_indices.argsort(axis=2)

        # # print(self.x)

        # x_coordinates, y_coordinates, _ = self.turbine_coordinates_array.T

        # x_center_of_rotation = (np.min(x_coordinates) + np.max(x_coordinates)) / 2
        # y_center_of_rotation = (np.min(y_coordinates) + np.max(y_coordinates)) / 2
        # # print(x_center_of_rotation)
        # # print(y_center_of_rotation)
        # # lkj

        # self.x = np.take_along_axis(self.x, self.unsorted_indices, axis=2)
        # self.y = np.take_along_axis(self.y, self.unsorted_indices, axis=2)
        # self.z = np.take_along_axis(self.z, self.unsorted_indices, axis=2)
        # # print(self.x)

        # self.x, self.y, self.z = self._rotated_grid(
        #     -1 * self.wind_directions,
        #     (x_center_of_rotation, y_center_of_rotation)
        # )
        # TODO figure out how to un-rotate grid for plotting after it has been solved
        # pass

    # def _rotated_grid(self, angle, center_of_rotation):
    #     """
    #     Rotate the discrete flow field grid.
    #     """
    #     angle = ((angle - 270) % 360 + 360) % 360
    #     # angle = np.reshape(angle, (len(angle), 1, 1))
    #     xoffset = self.x - center_of_rotation[0]
    #     yoffset = self.y - center_of_rotation[1]
    #     rotated_x = (
    #         xoffset * cosd(angle) - yoffset * sind(angle) + center_of_rotation[0]
    #     )
    #     rotated_y = (
    #         xoffset * sind(angle) + yoffset * cosd(angle) + center_of_rotation[1]
    #     )
    #     return rotated_x, rotated_y, self.z
