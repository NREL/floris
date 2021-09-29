from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from numpy import newaxis as na

from .utilities import Vec3, cosd, sind, tand


class Grid(ABC):
    def __init__(
        self,
        turbine_coordinates: List[Vec3],
        reference_turbine_diameter: float,
        reference_wind_height: float,
        grid_resolution: int,
    ) -> None:
        """
        Grid should establish domain bounds based on given criteria,
        and develop three arrays to contain components of the
        wind velocity.

        This could be generalized to any number of dimensions to be
        used by perhaps a turbulence field.

        The grid will have to be reestablished for each wind direction since the planform
        area of the farm will be different.
        """
        self.turbine_coordinates: List[Vec3] = turbine_coordinates
        self.reference_turbine_diameter: float = reference_turbine_diameter
        self.reference_wind_height: float = reference_wind_height
        self.grid_resolution: float = grid_resolution
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

    def rotate_fields(self, wd) -> None:
        # Find center of rotation
        x_center_of_rotation = (np.min(self.x) + np.max(self.x)) / 2
        y_center_of_rotation = (np.min(self.y) + np.max(self.y)) / 2

        angle = ((wd - 270) % 360 + 360) % 360
        # angle = (wd - 270) % 360 # Is this the same as above?

        # Rotate grid points
        x_offset = self.x - x_center_of_rotation
        y_offset = self.y - y_center_of_rotation
        mesh_x_rotated = (
            x_offset * cosd(angle) - y_offset * sind(angle) + x_center_of_rotation
        )
        mesh_y_rotated = (
            x_offset * sind(angle) + y_offset * cosd(angle) + y_center_of_rotation
        )

        # print(np.shape(mesh_x_rotated))
        # lkj
        self.x = mesh_x_rotated
        self.y = mesh_y_rotated

    @staticmethod
    def rotate_turbine_locations(
        coords: List[Vec3], wd
    ) -> Tuple[List[float], List[float]]:
        x_coord = [c.x1 for c in coords]
        y_coord = [c.x2 for c in coords]

        # Find center of rotation
        x_center_of_rotation = (np.min(x_coord) + np.max(x_coord)) / 2
        y_center_of_rotation = (np.min(y_coord) + np.max(y_coord)) / 2

        angle = ((wd - 270) % 360 + 360) % 360
        # angle = (wd - 270) % 360 # Is this the same as above?

        # Rotate turbine coordinates
        x_coord_offset = x_coord - x_center_of_rotation
        y_coord_offset = y_coord - y_center_of_rotation
        x_coord_rotated = (
            x_coord_offset * cosd(angle)
            - y_coord_offset * sind(angle)
            + x_center_of_rotation
        )
        y_coord_rotated = (
            x_coord_offset * sind(angle)
            + y_coord_offset * cosd(angle)
            + y_center_of_rotation
        )
        return x_coord_rotated, y_coord_rotated

    @property
    def n_turbines(self) -> int:
        return len(self.turbine_coordinates)


class TurbineGrid(Grid):
    def __init__(
        self,
        turbine_coordinates: List[Vec3],
        reference_turbine_diameter: float,
        reference_wind_height: float,
        grid_resolution: int,
    ) -> None:
        # establishes a data structure with grid on each turbine
        # the x,y,z points here are the grid points on the turbine swept area
        super().__init__(
            turbine_coordinates,
            reference_turbine_diameter,
            reference_wind_height,
            grid_resolution,
        )
        self.set_grid()

    def set_grid(self) -> None:
        """
        Create grid points at each turbine for each wind direction and wind speed in the simulation.
        This creates the underlying data structure for the calculation.

        arrays have shape (n turbines, m grid spanwise, m grid vertically)
        - dimension 1: each turbine
        - dimension 2: number of points in the spanwise direction (ngrid)
        - dimension 3: number of points in the vertical dimension (ngrid)

        For example
        - x is [n turbines, x-component of the points in the spanwise direction, x-component of the points in the vertical direction]
        - y is [n turbines, y-component of the points in the spanwise direction, y-component of the points in the vertical direction]

        The x array contains the actual locations in that direction.
        y and z values are displacements from the x coordinate in the y and z directions.
        That is to say that each turbine has a local coordinate system located at (x, )

        # -   **self.grid_resolution** (*int*, optional): The square root of the number
        #             of points to use on the turbine grid. This number will be
        #             squared so that the points can be evenly distributed.
        #             Defaults to 5.
        """
        # TODO: Where should we locate the coordinate system? Currently, its at
        # the foot of the turbine where the tower meets the ground.

        n_turbines = len(self.turbine_coordinates)

        # vector of size [1 x n_turbines]
        x_coordinates = np.expand_dims(
            np.array([c.x1 for c in self.turbine_coordinates]), axis=0
        )
        # y_coordinates = np.expand_dims(
        #     np.array([c.x2 for c in self.turbine_coordinates]), axis=0
        # )
        # z_coordinates = np.expand_dims(
        #     np.array([c.x3 for c in self.turbine_coordinates]), axis=0
        # )

        # -   **rloc** (*float, optional): A value, from 0 to 1, that determines
        #         the width/height of the grid of points on the rotor as a ratio of
        #         the rotor radius.
        #         Defaults to 0.5.
        radius_ratio = 0.5
        disc_area_radius = radius_ratio * self.reference_turbine_diameter / 2
        template_grid = np.ones(
            (n_turbines, self.grid_resolution * self.grid_resolution)
        )

        # Calculate the radial distance from the center of the turbine rotor
        disc_grid = np.linspace(
            -1 * disc_area_radius, disc_area_radius, self.grid_resolution
        )

        # Create the data for the turbine grids
        self.x = np.reshape(
            x_coordinates.T * template_grid,
            (n_turbines, self.grid_resolution, self.grid_resolution),
        )
        y_grid = np.zeros((n_turbines, self.grid_resolution, self.grid_resolution))
        z_grid = np.zeros((n_turbines, self.grid_resolution, self.grid_resolution))
        # print (y_coordinates.T)
        # print (disc_grid)

        for i, coord in enumerate(self.turbine_coordinates):
            #     # Save the indices of the flow field points for this turbine
            #     # Create the grids
            #     x_grid[i] = coord.x1
            y_grid[i] = coord.x2 + disc_grid
            z_grid[i] = coord.x3 + disc_grid

        self.y = y_grid
        self.z = z_grid


class FlowFieldGrid(Grid):
    """
    Primarily used by the Curl model and for visualization
    """

    def __init__(
        self,
        turbine_coordinates: List[Vec3],
        reference_turbine_diameter: float,
        reference_wind_height: float,
        grid_resolution: Vec3,
    ) -> None:

        # the x,y points are a regular grid based on given domain bounds

        super().__init__(
            turbine_coordinates,
            reference_turbine_diameter,
            reference_wind_height,
            grid_resolution,
        )

        self.set_bounds()
        self.set_grid()

    def set_bounds(self) -> None:
        # TODO: Should this be called "compute_bounds?"
        #   anything set_ could require an argument to set a value
        #   other functions that set variables based on previous inputs could be "compute_"
        #   anything that returns values, even if they are computed on the fly, could be get_ (instead of @property)
        """
        Calculates the domain bounds for the current wake model. The bounds
        are calculated based on preset extents from the
        given layout. The bounds consist of the minimum and maximum values
        in the x-, y-, and z-directions.

        If the Curl model is used, the predefined bounds are always set.
        """
        # For the curl model, bounds are hard coded
        coords = self.turbine_coordinates
        x = [coord.x1 for coord in coords]
        y = [coord.x2 for coord in coords]
        eps = 0.1
        self.xmin = min(x) - 2 * self.reference_turbine_diameter
        self.xmax = max(x) + 10 * self.reference_turbine_diameter
        self.ymin = min(y) - 2 * self.reference_turbine_diameter
        self.ymax = max(y) + 2 * self.reference_turbine_diameter
        self.zmin = 0 + eps
        self.zmax = 6 * self.reference_wind_height

    def set_grid(self) -> None:
        """
        Create a structured grid for the entire flow field domain.
        resolution: Vec3
        """
        x_points = np.linspace(self.xmin, self.xmax, int(self.grid_resolution.x1))
        y_points = np.linspace(self.ymin, self.ymax, int(self.grid_resolution.x2))
        z_points = np.linspace(self.zmin, self.zmax, int(self.grid_resolution.x3))
        self.x, self.y, self.z = np.meshgrid(
            x_points, y_points, z_points, indexing="ij"
        )
