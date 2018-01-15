# Copyright 2017 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
from .visualization_manager import VisualizationManager
from .coordinate import Coordinate

class FlowField():
    """
    FlowField is at the core of the FLORIS package. This class handles the domain
    creation and initialization and computes the flow field based on the input
    wake model and turbine map. It also contains helper functions for quick flow
    field visualization.
        
    inputs:
        wind_speed: float - atmospheric condition

        wind_direction - atmospheric condition
        
        wind_shear - atmospheric condition
        
        wind_veer - atmospheric condition
        
        turbulence_intensity - atmospheric condition
        
        wake: Wake - used to calculate the flow field
        
        wake_combination: WakeCombination - used to combine turbine wakes into the flow field
        
        turbine_map: TurbineMap - locates turbines in space

    outputs:
        self: FlowField - an instantiated FlowField object
    """

    def __init__(self,
                 wind_speed,
                 wind_direction,
                 wind_shear,
                 wind_veer,
                 turbulence_intensity,
                 wake,
                 wake_combination,
                 turbine_map):

        super().__init__()

        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.wind_shear = wind_shear
        self.wind_veer = wind_veer
        self.turbulence_intensity = turbulence_intensity
        self.wake = wake
        self.wake_combination = wake_combination
        self.turbine_map = turbine_map
        
        # initialize derived attributes and constants
        self.max_diameter = max(
            [turbine.rotor_diameter for turbine in self.turbine_map.turbines])
        self.hub_height = self.turbine_map.turbines[0].hub_height
        self.grid_resolution = Coordinate(100, 100, 25)
        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = self._set_domain_bounds()
        self.x, self.y, self.z = self._discretize_domain()
        self.initial_flowfield = self._initial_flowfield()
        self.u_field = self._initial_flowfield()

        self.viz_manager = VisualizationManager()

    def _set_domain_bounds(self):
        coords = self.turbine_map.coords
        x = [coord.x for coord in coords]
        y = [coord.y for coord in coords]
        eps = 0.1
        xmin = min(x) - 2 * self.max_diameter
        xmax = max(x) + 10 * self.max_diameter
        ymin = min(y) - 2 * self.max_diameter
        ymax = max(y) + 2 * self.max_diameter
        zmin = 0 + eps 
        zmax = 2 * self.hub_height
        return xmin, xmax, ymin, ymax, zmin, zmax

    def _discretize_domain(self):
        x = np.linspace(self.xmin, self.xmax, self.grid_resolution.x)
        y = np.linspace(self.ymin, self.ymax, self.grid_resolution.y)
        z = np.linspace(self.zmin, self.zmax, self.grid_resolution.z)
        return np.meshgrid(x, y, z, indexing="ij")

    def _map_coordinate_to_index(self, coord):
        """
        """
        xi = max(0, int(self.grid_resolution.x * (coord.x - self.xmin - 1) \
            / (self.xmax - self.xmin)))
        yi = max(0, int(self.grid_resolution.y * (coord.y - self.ymin - 1) \
            / (self.ymax - self.ymin)))
        zi = max(0, int(self.grid_resolution.z * (coord.z - self.zmin - 1) \
            / (self.zmax - self.zmin)))
        return xi, yi, zi

    def _field_value_at_coord(self, target_coord, field):
        xi, yi, zi = self._map_coordinate_to_index(target_coord)
        return field[xi, yi, zi]

    def _initial_flowfield(self):
        turbines = self.turbine_map.turbines
        max_diameter = max([turbine.rotor_diameter for turbine in turbines])
        return self.wind_speed * (self.z / self.hub_height)**self.wind_shear

    def _compute_turbine_velocity_deficit(self, x, y, z, turbine, coord, deflection, wake, flowfield):
        velocity_function = self.wake.get_velocity_function()
        return velocity_function(x, y, z, turbine, coord, deflection, wake, flowfield)

    def _compute_turbine_wake_deflection(self, x, y, turbine, coord, flowfield):
        deflection_function = self.wake.get_deflection_function()
        return deflection_function(x, y, turbine, coord, flowfield)

    def _rotated_grid(self, angle, center_of_rotation):
        xoffset = self.x - center_of_rotation.x
        yoffset = self.y - center_of_rotation.y
        rotated_x = xoffset * \
            np.cos(angle) - yoffset * \
            np.sin(angle) + center_of_rotation.x
        rotated_y = xoffset * \
            np.sin(angle) + yoffset * \
            np.cos(angle) + center_of_rotation.y
        return rotated_x, rotated_y, self.z

    def _calculate_area_overlap(self, wake_velocities, freestream_velocities, turbine):
        # compute wake overlap based on the number of points that are not freestream velocity, i.e. affected by the wake
        count = np.sum(freestream_velocities - wake_velocities <= 0.05)
        return (turbine.grid_point_count - count) / turbine.grid_point_count

    # Public methods

    def calculate_wake(self):

        # initialize turbulence intensity at every turbine (seems sloppy)
        for coord, turbine in self.turbine_map.items():
            turbine.TI = self.turbulence_intensity

        # rotate the discrete grid and turbine map
        center_of_rotation = Coordinate(
            np.mean(np.unique(self.x)), np.mean(np.unique(self.y)))

        rotated_x, rotated_y, rotated_z = self._rotated_grid(
            self.wind_direction, center_of_rotation)

        rotated_map = self.turbine_map.rotated(
            self.wind_direction, center_of_rotation)

        # sort the turbine map
        sorted_map = rotated_map.sorted_in_x_as_list()

        # calculate the velocity deficit and wake deflection on the mesh
        u_wake = np.zeros(self.u_field.shape)
        for coord, turbine in sorted_map:

            # update the turbine based on the velocity at its hub
            # local_deficit = self._field_velocity_at_coord(coord, u_wake)
            # turbine.update_quantities(self.wind_speed, self.wind_speed - local_deficit, self.wind_shear,self)
            turbine.update_quantities(u_wake, coord, self, rotated_x, rotated_y, rotated_z)
            
            # get the wake deflecton field
            deflection = self._compute_turbine_wake_deflection(rotated_x, rotated_y, turbine, coord, self)

            # get the velocity deficit accounting for the deflection
            turb_wake = self._compute_turbine_velocity_deficit(
                rotated_x, rotated_y, rotated_z, turbine, coord, deflection, self.wake, self)

            # compute area overlap of wake on other turbines and update downstream turbine turbulence intensities

            if self.wake.velocity_model == 'gauss':
                for coord_ti, _ in sorted_map:

                    if coord_ti.x > coord.x:
                        turbine_ti = rotated_map[coord_ti]

                        # only assess the effects of the current wake
                        wake_velocities = turbine_ti._calculate_swept_area_velocities(self, self.initial_flowfield - turb_wake, 
                                            coord_ti, rotated_x, rotated_y, rotated_z)
                        freestream_velocities = turbine_ti._calculate_swept_area_velocities(self, self.initial_flowfield, 
                                            coord_ti, rotated_x, rotated_y, rotated_z)

                        area_overlap = self._calculate_area_overlap(wake_velocities, freestream_velocities, turbine)

                        if area_overlap > 0.0:
                            turbine_ti.TI = turbine_ti._calculate_turbulence_intensity(self,self.wake,coord_ti,coord,turbine)

            # combine this turbine's wake into the full wake field
            u_wake = self.wake_combination.combine(u_wake, turb_wake)

        # apply the velocity deficit field to the freestream
        self.u_field = self.initial_flowfield - u_wake

    # Visualization

    def _add_z_plane(self, percent_height=0.5):
        plane = int(self.grid_resolution.z * percent_height)
        self.viz_manager.plot_constant_z(
            self.x[:, :, plane], self.y[:, :, plane], self.u_field[:, :, plane])
        for coord, turbine in self.turbine_map.items():
            self.viz_manager.add_turbine_marker(turbine, coord, self.wind_direction)

    def _add_y_plane(self, percent_height=0.5):
        plane = int(self.grid_resolution.y * percent_height)
        self.viz_manager.plot_constant_y(
            self.x[:, plane, :], self.z[:, plane, :], self.u_field[:, plane, :])

    def _add_x_plane(self, percent_height=0.5):
        plane = int(self.grid_resolution.x * percent_height)
        self.viz_manager.plot_constant_x(
            self.y[plane, :, :], self.z[plane, :, :], self.u_field[plane, :, :])

    def plot_z_planes(self, planes):
        for p in planes:
            self._add_z_plane(p)
        self.viz_manager.show()

    def plot_y_planes(self, planes):
        for p in planes:
            self._add_y_plane(p)
        self.viz_manager.show()

    def plot_x_planes(self, planes):
        for p in planes:
            self._add_x_plane(p)
        self.viz_manager.show()
