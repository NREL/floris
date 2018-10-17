# Copyright 2017 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from .coordinate import Coordinate
import matplotlib.pyplot as plt
import numpy as np

class VisualizationManager():
    """
    The VisualizationManager handles all of the lower level visualization instantiation
    and data management. Currently, it produces 2D matplotlib plots for a given plane
    of data.

    IT IS IMPORTANT to note that this class should be treated as a singleton. That is,
    only one instance of this class should exist.
    """

    def __init__(self, flowfield, grid_resolution=(100, 100, 25)):
        self.figure_count = 0
        self.flowfield = flowfield
        self.grid_resolution = Coordinate(grid_resolution[0], grid_resolution[1], grid_resolution[2])
        self._initialize_flowfield_for_plotting()

    # General plotting functions

    def _set_texts(self, plot_title, horizontal_axis_title, vertical_axis_title):
        fontsize = 15
        plt.title(plot_title, fontsize=fontsize)
        plt.xlabel(horizontal_axis_title, fontsize=fontsize)
        plt.ylabel(vertical_axis_title, fontsize=fontsize)

    def _set_colorbar(self, label):
        cb = plt.colorbar()
        cb.set_label(label)
        cb.ax.tick_params(labelsize=15)

    def _set_axis(self):
        plt.axis('equal')
        plt.tick_params(which='both', labelsize=15)

    def _new_figure(self):
        plt.figure()
        self.figure_count += 1

    def _new_filled_contour(self, mesh1, mesh2, data):
        self._new_figure()
        vmax = np.amax(data)
        plt.contourf(mesh1, mesh2, data, 50,
                     cmap='viridis', vmin=0, vmax=vmax)

    def _plot_constant_plane(self,
                             mesh1,
                             mesh2,
                             data,
                             title,
                             xlabel,
                             ylabel,
                             colorbar=True,
                             colorbar_label=''):
        self._new_filled_contour(mesh1, mesh2, data)
        self._set_texts(title, xlabel, ylabel)
        if colorbar:
            self._set_colorbar(colorbar_label)
        self._set_axis()

    # FLORIS-specific data manipulation and plotting
    def _initialize_flowfield_for_plotting(self):
        self.flowfield.grid_resolution = self.grid_resolution
        self.flowfield.xmin, self.flowfield.xmax, self.flowfield.ymin, self.flowfield.ymax, self.flowfield.zmin, self.flowfield.zmax = self._set_domain_bounds()
        self.flowfield.x, self.flowfield.y, self.flowfield.z = self._discretize_freestream_domain()
        self.flowfield.initial_flowfield = self.flowfield._initial_flowfield()
        self.flowfield.u_field = self.flowfield._initial_flowfield()
        for turbine in self.flowfield.turbine_map.turbines:
            turbine.plotting = True
        self.flowfield.calculate_wake()

    def _discretize_freestream_domain(self):
        """
        Generate a structured grid for the entire flow field domain.
        """
        x = np.linspace(self.flowfield.xmin, self.flowfield.xmax, self.flowfield.grid_resolution.x)
        y = np.linspace(self.flowfield.ymin, self.flowfield.ymax, self.flowfield.grid_resolution.y)
        z = np.linspace(self.flowfield.zmin, self.flowfield.zmax, self.flowfield.grid_resolution.z)
        return np.meshgrid(x, y, z, indexing='ij')

    def _set_domain_bounds(self):
        coords = self.flowfield.turbine_map.coords
        x = [coord.x for coord in coords]
        y = [coord.y for coord in coords]
        eps = 0.1
        xmin = min(x) - 5 * self.flowfield.max_diameter
        xmax = max(x) + 10 * self.flowfield.max_diameter
        ymin = min(y) - 5 * self.flowfield.max_diameter
        ymax = max(y) + 5 * self.flowfield.max_diameter
        zmin = 0 + eps
        zmax = 2 * self.flowfield.hub_height
        return xmin, xmax, ymin, ymax, zmin, zmax

    def _add_turbine_marker(self, turbine, coords, wind_direction):
        a = Coordinate(coords.x, coords.y - turbine.rotor_radius)
        b = Coordinate(coords.x, coords.y + turbine.rotor_radius)
        a.rotate_z(turbine.yaw_angle - wind_direction, coords.as_tuple())
        b.rotate_z(turbine.yaw_angle - wind_direction, coords.as_tuple())
        plt.plot([a.xprime, b.xprime], [a.yprime, b.yprime], 'k', linewidth=1)

    def _plot_constant_z(self, xmesh, ymesh, data, **kwargs):
        self._plot_constant_plane(
            xmesh, ymesh, data, 'z plane', 'x (m)', 'y (m)', colorbar_label='Flow speed (m/s)', **kwargs)

    def _plot_constant_y(self, xmesh, zmesh, data, **kwargs):
        self._plot_constant_plane(
            xmesh, zmesh, data, 'y plane', 'x (m)', 'z (m)', colorbar_label='Flow speed (m/s)', **kwargs)

    def _plot_constant_x(self, ymesh, zmesh, data, **kwargs):
        self._plot_constant_plane(
            ymesh, zmesh, data, 'x plane', 'y (m)', 'z (m)', colorbar_label='Flow speed (m/s)', **kwargs)

    def _add_z_plane(self, percent_height=0.5, **kwargs):
        plane = int(self.flowfield.grid_resolution.z * percent_height)
        self._plot_constant_z(
            self.flowfield.x[:, :, plane],
            self.flowfield.y[:, :, plane],
            self.flowfield.u_field[:, :, plane],
            **kwargs)
        for coord, turbine in self.flowfield.turbine_map.items():
            self._add_turbine_marker(
                turbine, coord, self.flowfield.wind_direction)

    def _add_y_plane(self, percent_height=0.5, **kwargs):
        plane = int(self.flowfield.grid_resolution.y * percent_height)
        self._plot_constant_y(
            self.flowfield.x[:, plane, :],
            self.flowfield.z[:, plane, :],
            self.flowfield.u_field[:, plane, :],
            **kwargs)

    def _add_x_plane(self, percent_height=0.5, **kwargs):
        plane = int(self.flowfield.grid_resolution.x * percent_height)
        self._plot_constant_x(
            self.flowfield.y[plane, :, :],
            self.flowfield.z[plane, :, :],
            self.flowfield.u_field[plane, :, :],
            **kwargs)

    def plot_z_planes(self, planes, **kwargs):
        for p in planes:
            self._add_z_plane(p, **kwargs)

    def plot_y_planes(self, planes, **kwargs):
        for p in planes:
            self._add_y_plane(p, **kwargs)

    def plot_x_planes(self, planes, **kwargs):
        for p in planes:
            self._add_x_plane(p, **kwargs)

    def show(self):
        plt.show()

    # def _map_coordinate_to_index(self, coord):
    #     xi = max(0, int(self.grid_resolution.x * (coord.x - self.xmin - 1) \
    #         / (self.xmax - self.xmin)))
    #     yi = max(0, int(self.grid_resolution.y * (coord.y - self.ymin - 1) \
    #         / (self.ymax - self.ymin)))
    #     zi = max(0, int(self.grid_resolution.z * (coord.z - self.zmin - 1) \
    #         / (self.zmax - self.zmin)))
    #     return xi, yi, zi

    # def _field_value_at_coord(self, target_coord, field):
    #     xi, yi, zi = self._map_coordinate_to_index(target_coord)
    #     return field[xi, yi, zi]
