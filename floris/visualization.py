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

    def __init__(self, flow_field, grid_resolution=(100, 100, 25)):
        self.figure_count = 0
        self.flow_field = flow_field
        self.grid_resolution = Coordinate(grid_resolution[0], grid_resolution[1], grid_resolution[2])
        self._initialize_flow_field_for_plotting()

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
    def _initialize_flow_field_for_plotting(self):
        if self.flow_field.wake.velocity_model.type_string != 'curl':
            self.flow_field.grid_resolution = self.grid_resolution
            self.flow_field.xmin, self.flow_field.xmax, self.flow_field.ymin, self.flow_field.ymax, self.flow_field.zmin, self.flow_field.zmax = self._set_domain_bounds()
            self.flow_field.x, self.flow_field.y, self.flow_field.z = self._discretize_freestream_domain()
            self.flow_field.initial_flow_field, self.flow_field.v_initial, self.flow_field.w_initial = self.flow_field.initialize_flow_field()
            self.flow_field.u_field, self.flow_field.v, self.flow_field.w = self.flow_field.initialize_flow_field()
            for turbine in self.flow_field.turbine_map.turbines:
                turbine.plotting = True
            self.flow_field.calculate_wake()

    def _discretize_freestream_domain(self):
        """
        Generate a structured grid for the entire flow field domain.
        """
        x = np.linspace(self.flow_field.xmin, self.flow_field.xmax, self.flow_field.grid_resolution.x)
        y = np.linspace(self.flow_field.ymin, self.flow_field.ymax, self.flow_field.grid_resolution.y)
        z = np.linspace(self.flow_field.zmin, self.flow_field.zmax, self.flow_field.grid_resolution.z)
        return np.meshgrid(x, y, z, indexing='ij')

    def _set_domain_bounds(self):
        coords = self.flow_field.turbine_map.coords
        x = [coord.x for coord in coords]
        y = [coord.y for coord in coords]
        eps = 0.1
        xmin = min(x) - 5 * self.flow_field.max_diameter
        xmax = max(x) + 10 * self.flow_field.max_diameter
        ymin = min(y) - 5 * self.flow_field.max_diameter
        ymax = max(y) + 5 * self.flow_field.max_diameter
        zmin = 0 + eps
        zmax = 2 * self.flow_field.hub_height
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
        plane = int(self.flow_field.grid_resolution.z * percent_height)
        self._plot_constant_z(
            self.flow_field.x[:, :, plane],
            self.flow_field.y[:, :, plane],
            self.flow_field.u_field[:, :, plane],
            **kwargs)
        for coord, turbine in self.flow_field.turbine_map.items():
            self._add_turbine_marker(
                turbine, coord, self.flow_field.wind_direction)

    def _add_y_plane(self, percent_height=0.5, **kwargs):
        plane = int(self.flow_field.grid_resolution.y * percent_height)
        self._plot_constant_y(
            self.flow_field.x[:, plane, :],
            self.flow_field.z[:, plane, :],
            self.flow_field.u_field[:, plane, :],
            **kwargs)

    def _add_x_plane(self, percent_height=0.5, **kwargs):
        plane = int(self.flow_field.grid_resolution.x * percent_height)
        self._plot_constant_x(
            self.flow_field.y[plane, :, :],
            self.flow_field.z[plane, :, :],
            self.flow_field.u_field[plane, :, :],
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
