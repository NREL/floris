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

import attr
import numpy as np
import numpy.typing as npt

from floris.utilities import Vec3, cosd, sind, attrs_array_converter
from floris.utilities import rotate_coordinates_rel_west


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
        grid_resolution (:py:obj:`int` | :py:obj:`Iterable(int,)`): Grid resolution specific to each grid type
    """

    # TODO: We'll want to consider how this expands to multiple turbine types
    turbine_coordinates: list[Vec3] = attr.ib(on_setattr=attr.setters.validate)
    reference_turbine_diameter: float
    grid_resolution: int | Iterable = attr.ib(on_setattr=attr.setters.validate)
    wind_directions: NDArrayFloat = attr.ib(converter=attrs_array_converter, on_setattr=(attr.setters.convert, attr.setters.validate))
    wind_speeds: NDArrayFloat = attr.ib(converter=attrs_array_converter, on_setattr=(attr.setters.convert, attr.setters.validate))

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
            raise TypeError("'turbine_coordinates' must be `Vec3` objects.")

        self.n_turbines = len(value)

    @wind_speeds.validator
    def wind_speeds_validator(self, instance: attr.Attribute, value: NDArrayFloat) -> None:
        """Using the validator method to keep the `n_wind_speeds` attribute up to date."""
        self.n_wind_speeds = value.size

    @wind_directions.validator
    def wind_directions_validator(self, instance: attr.Attribute, value: NDArrayFloat) -> None:
        """Using the validator method to keep the `n_wind_directions` attribute up to date."""
        self.n_wind_directions = value.size

    @grid_resolution.validator
    def grid_resolution_validator(self, instance: attr.Attribute, value: int | Iterable) -> None:
        """Check that grid resolution is given as int or Vec3 with int components."""
        if isinstance(value, int):
            return
        elif isinstance(value, Iterable):
            assert type(value[0]) is int
            assert type(value[1]) is int
            assert type(value[2]) is int
        else:
            raise TypeError("`grid_resolution` must be of type int or Iterable(int,)")

    @abstractmethod
    def set_grid(self) -> None:
        raise NotImplementedError("Grid.set_grid")


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
            )
        )

        # Calculate the radial distance from the center of the turbine rotor
        disc_grid = np.linspace(-1 * disc_area_radius, disc_area_radius, self.grid_resolution)

        # Construct the turbine grids
        # Here, they are already rotated to the correct orientation for each wind direction
        _x = x[:, :, :, None, None] * template_grid
        # [:, None] on disc_grid below is effectively a transpose of this vector; compare with disc_grid.reshape(1,-1).T
        _y = y[:, :, :, None, None] + template_grid * ( disc_grid[:, None] * np.ones((self.grid_resolution, self.grid_resolution)) )
        _z = z[:, :, :, None, None] + template_grid * ( disc_grid * np.ones((self.grid_resolution, self.grid_resolution)) )

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


class FlowFieldGrid(Grid):
    """
    Args:
        grid_resolution (`Vec3`): The number of grid points to be created in each direction.
        turbine_coordinates (`list[Vec3]`): The collection of turbine coordinate (`Vec3`) objects.
        reference_turbine_diameter (:py:obj:`float`): The reference turbine's rotor diameter.
        grid_resolution (:py:obj:`int`): The number of points on each turbine
    """

    grid_resolution: Iterable
    xmin: float = attr.ib(init=False)
    xmax: float = attr.ib(init=False)
    ymin: float = attr.ib(init=False)
    ymax: float = attr.ib(init=False)
    zmin: float = attr.ib(init=False)
    zmax: float = attr.ib(init=False)

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
        xmin = min(x) - 2 * self.reference_turbine_diameter
        xmax = max(x) + 10 * self.reference_turbine_diameter
        ymin = min(y) - 2 * self.reference_turbine_diameter
        ymax = max(y) + 2 * self.reference_turbine_diameter
        zmin = 0 + eps
        zmax = 6 * max(z)

        x_points, y_points, z_points = np.meshgrid(
            np.linspace(xmin, xmax, int(self.grid_resolution[0])),
            np.linspace(ymin, ymax, int(self.grid_resolution[1])),
            np.linspace(zmin, zmax, int(self.grid_resolution[2])),
            indexing="ij"
        )

        self.x = x_points[None, None, :, :, :]
        self.y = y_points[None, None, :, :, :]
        self.z = z_points[None, None, :, :, :]

    def finalize(self):
        pass
