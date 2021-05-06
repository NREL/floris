
import numpy as np
from .utilities import Vec3, cosd, sind, tand
from typing import List


class Grid():
    def __init__(self) -> None:
        """
        Grid should establish domain bounds based on given criteria,
        and develop three arrays to contain components of the
        wind velocity.

        This could be generalized to any number of dimensions to be
        used by perhaps a turbulence field.
        """
        # x are the locations in space in the primary direction (typically the direction of the wind)
        # y are the locations in space in the lateral direction
        # z are the locations in space in the vertical direction
        # u are the velocity components at each point in space
        # v are the velocity components at each point in space
        # w are the velocity components at each point in space
        # all of these arrays are the same size
        pass

    def set_bounds(self):
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

    def rotated(self, angle, center_of_rotation):
        """
        Rotate the discrete flow field grid.
        """
        xoffset = self.x - center_of_rotation.x1
        yoffset = self.y - center_of_rotation.x2
        rotated_x = (
            xoffset * cosd(angle) - yoffset * sind(angle) + center_of_rotation.x1
        )
        rotated_y = (
            xoffset * sind(angle) + yoffset * cosd(angle) + center_of_rotation.x2
        )
        return rotated_x, rotated_y, self.z

    def _update_grid(self, x_grid_i, y_grid_i, wind_direction_i, x1, x2):
        xoffset = x_grid_i - x1
        yoffset = y_grid_i.T - x2
        wind_cos = cosd(-wind_direction_i)
        wind_sin = sind(-wind_direction_i)

        x_grid_i = xoffset * wind_cos - yoffset * wind_sin + x1
        y_grid_i = yoffset * wind_cos + xoffset * wind_sin + x2
        return x_grid_i, y_grid_i

    def get_bounds(self) -> List[float]:
        """
        The minimum and maximum values of the bounds of the computational domain.
        """
        return [self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax]


class TurbineGrid(Grid):
    def __init__(self, turbine_coordinates: List[Vec3], reference_turbine_diameter: float, reference_wind_height: float, grid_resolution: float) -> None:
        # establishes a data structure with grid on each turbine
        # the x,y points here are the turbine locations

        self.turbine_coordinates: List[Vec3] = turbine_coordinates
        self.reference_turbine_diameter: float = reference_turbine_diameter
        self.reference_wind_height: float = reference_wind_height
        self.grid_resolution: Vec3 = grid_resolution

        super().__init__()

        self.set_bounds()
        self.set_grid()

    def set_grid(self) -> None:
        """
        Create grid points at each turbine

        arrays have shape (n turbines, n grid, n grid)
        - dimension 1: each turbine
        - dimension 2: number of points in the spanwise direction (ngrid)
        - dimension 3: number of points in the vertical direction (ngrid)

        # -   **self.grid_resolution** (*int*, optional): The square root of the number
        #             of points to use on the turbine grid. This number will be
        #             squared so that the points can be evenly distributed.
        #             Defaults to 5.
        """
        n_turbines = len(self.turbine_coordinates)

        # vector of size [1 x n_turbines]
        x_coordinates = np.expand_dims(np.array([c.x1 for c in self.turbine_coordinates]), axis=0)
        y_coordinates = np.expand_dims(np.array([c.x2 for c in self.turbine_coordinates]), axis=0)
        z_coordinates = np.expand_dims(np.array([c.x3 for c in self.turbine_coordinates]), axis=0)

        # -   **rloc** (*float, optional): A value, from 0 to 1, that determines
        #         the width/height of the grid of points on the rotor as a ratio of
        #         the rotor radius.
        #         Defaults to 0.5.
        radius_ratio = 0.5
        disc_area_radius = radius_ratio * self.reference_turbine_diameter / 2

        template_grid = np.ones((n_turbines, self.grid_resolution * self.grid_resolution))
        disc_grid = np.linspace(-1 * disc_area_radius, disc_area_radius, self.grid_resolution)

        # Create the data for the turbine grids
        self.x = np.reshape(x_coordinates.T * template_grid, (n_turbines, self.grid_resolution, self.grid_resolution) )
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

            # ?
            # x_grid[i], y_grid[i] = self._update_grid(x_grid[i], y_grid[i], self.wind_map.turbine_wind_direction[i], x1, x2)``
        self.y = y_grid
        self.z = z_grid

    def compute_initialized_domain(self):
        """
        Establish the layout of grid points for the flow field domain and
        calculate initial values at these points.

        1) Initializing a non-curl model (gauss, multizone), using a call to _discretize_turbine_domain
        """
        self.x, self.y, self.z = self._discretize_turbine_domain()

        # set grid point locations in wind_map
        self.wind_map.grid_layout = (self.x, self.y)

        # interpolate for initial values of flow field grid
        self.wind_map.calculate_turbulence_intensity(grid=True)
        self.wind_map.calculate_wind_direction(grid=True)
        self.wind_map.calculate_wind_speed(grid=True)


class FlowFieldGrid(Grid):
    """
    Primarily used by the Curl model and for visualization
    """
    def __init__(self, turbine_coordinates: List[Vec3], reference_turbine_diameter: float, reference_wind_height: float, grid_resolution: Vec3) -> None:

        # the x,y points are a regular grid based on given domain bounds

        self.turbine_coordinates: List[Vec3] = turbine_coordinates
        self.reference_turbine_diameter: float = reference_turbine_diameter
        self.reference_wind_height: float = reference_wind_height
        self.grid_resolution: Vec3 = grid_resolution

        super().__init__()

        self.set_bounds()
        self.set_grid()

    def set_grid(self) -> None:
        """
        Create a structured grid for the entire flow field domain.
        resolution: Vec3
        """
        x_points = np.linspace(self.xmin, self.xmax, int(self.grid_resolution.x1))
        y_points = np.linspace(self.ymin, self.ymax, int(self.grid_resolution.x2))
        z_points = np.linspace(self.zmin, self.zmax, int(self.grid_resolution.x3))
        self.x, self.y, self.z = np.meshgrid(x_points, y_points, z_points, indexing="ij")
