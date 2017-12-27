"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from .BaseObject import BaseObject
from .VisualizationManager import VisualizationManager
from .Coordinate import Coordinate
import numpy as np
import copy

class FlowField(BaseObject):
    """
        Describe FF here
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

        self.grid_x_resolution = 200
        self.grid_y_resolution = 200
        self.grid_z_resolution = 50

        self.x, self.y, self.z = self._discretize_domain()
        self.u_field = self._constant_flowfield(self.wind_speed)
        self.initial_flowfield = self._initial_flowfield()

        self.viz_manager = VisualizationManager()

    def _discretize_domain(self):
        coords = [coord for coord, _ in self.turbine_map.items()]
        x = [coord.x for coord in coords]
        y = [coord.y for coord in coords]
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)

        turbines = [turbine for _, turbine in self.turbine_map.items()]
        maxDiameter = max([turbine.rotor_diameter for turbine in turbines])
        hub_height = turbines[0].hub_height

        x = np.linspace(xmin - 5 * maxDiameter, xmax + 10 * maxDiameter, self.grid_x_resolution)
        y = np.linspace(ymin - 5 * maxDiameter, ymax + 2 * maxDiameter, self.grid_y_resolution)
        z = np.linspace(0.1, 2 * hub_height, self.grid_z_resolution)
        return np.meshgrid(x, y, z, indexing="xy")

    def _field_velocity_at_coord(self, target_coord, field):
        coords = [coord for coord, _ in self.turbine_map.items()]
        x = [coord.x for coord in coords]
        y = [coord.y for coord in coords]
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)

        turbines = [turbine for _, turbine in self.turbine_map.items()]
        maxDiameter = max([turbine.rotor_diameter for turbine in turbines])

        #x_range = (xmin - 5 * maxDiameter, xmax + 10 * maxDiameter, self.grid_x_resolution)
        #y_range = (ymin - 5 * maxDiameter, ymax + 2 * maxDiameter, self.grid_y_resolution)
        x_range = (np.min(self.x), np.max(self.x), self.grid_x_resolution)
        y_range = (np.min(self.y), np.max(self.y), self.grid_y_resolution)

        dx = (x_range[1] - x_range[0]) / self.grid_x_resolution
        dy = (y_range[1] - y_range[0]) / self.grid_y_resolution

        xindex = int((target_coord.x + maxDiameter) / dx) + 1
        yindex = int((target_coord.y + maxDiameter) / dy)

        return field[yindex, xindex, 25]

    #def _grid_velocities(self, turbine, coord):

    #    # extract velocities at each of the grid points



    def _constant_flowfield(self, constant_value):
        xmax, ymax, zmax = self.x.shape[0], self.y.shape[1], self.z.shape[2]
        return np.full((xmax, ymax, zmax), constant_value)

    def _initial_flowfield(self):

        turbines = [turbine for _, turbine in self.turbine_map.items()]
        maxDiameter = max([turbine.rotor_diameter for turbine in turbines])
        hub_height = turbines[0].hub_height

        return self.wind_speed*(self.z/hub_height)**self.wind_shear

    def _compute_turbine_velocity_deficit(self, x, y, z, turbine, coord, deflection, wake, flowfield):
        velocity_function = self.wake.get_velocity_function()
        return velocity_function(x, y, z, turbine, coord, deflection, wake, flowfield)

    def _compute_turbine_wake_deflection(self, x, y, turbine, coord, flowfield):
        deflection_function = self.wake.get_deflection_function()
        return deflection_function(x, y, turbine, coord, flowfield)

    def _rotate_coordinates(self):

        # this rotates the turbine coordinates such that they are now in the frame of reference of the 270 degree wind direction.
        # this makes computing wakes and wake overlap much simpler

        rotation_angle = (self.wind_direction - 270.)*np.pi/180.
        xCenter = np.mean(np.unique(self.x))
        yCenter = np.mean(np.unique(self.y))

        rotated_x = (self.x-xCenter)*np.cos(rotation_angle) - (self.y-yCenter)*np.sin(rotation_angle) + xCenter 
        rotated_y = (self.x-xCenter)*np.sin(rotation_angle) + (self.y-yCenter)*np.cos(rotation_angle) + yCenter 
        rotated_z = self.z

        rotated_map = dict()
        for coord,turbine in self.turbine_map.items():
            x_rotated = (coord.x-xCenter)*np.cos(rotation_angle) - (coord.y-yCenter)*np.sin(rotation_angle)
            y_rotated = (coord.x-xCenter)*np.sin(rotation_angle) + (coord.y-yCenter)*np.cos(rotation_angle)
            rotated_map[Coordinate(x_rotated+xCenter,y_rotated+yCenter)] = turbine

        return rotated_map, rotated_x, rotated_y, rotated_z

    def _calculate_area_overlap(self, wake_velocities, freestream_velocities, turbine):

        # compute wake overlap based on the number of points that are not freestream velocity, i.e. affected by the wake
        count = np.sum(freestream_velocities - wake_velocities <= 0.05)

        return (turbine.grid_point_count-count)/turbine.grid_point_count

    # Public methods

    def calculate_wake(self):

        # initialize turbulence intensity at every turbine (seems sloppy)
        for coord,turbine in self.turbine_map.items():
            turbine.TI = self.turbulence_intensity

        # rotate and sort the turbine coordinates
        rotated_map, rotated_x, rotated_y, rotated_z = self._rotate_coordinates()
        sorted_coords = sorted(rotated_map, key=lambda coord:coord.x)

        # calculate the velocity deficit and wake deflection on the mesh
        u_wake = np.zeros(self.u_field.shape)

        for coord in sorted_coords:

            # assign turbine based on coordinates
            turbine = rotated_map[coord]

            # update the turbine based on the velocity at its hub
            local_deficit = self._field_velocity_at_coord(coord, u_wake)
            #turbine.update_quantities(self.wind_speed, self.wind_speed - local_deficit, self.wind_shear,self)
            turbine.update_quantities(u_wake, coord, rotated_map, self, rotated_x, rotated_y, rotated_z)
            
            # get the wake deflecton field
            deflection = self._compute_turbine_wake_deflection(rotated_x, rotated_y, turbine, coord, self)

            # get the velocity deficit accounting for the deflection
            turb_wake = self._compute_turbine_velocity_deficit(
                rotated_x, rotated_y, rotated_z, turbine, coord, deflection, self.wake, self)

            # compute area overlap of wake on other turbines and update downstream turbine turbulence intensities

            if self.wake.velocity_model == 'gauss':
                for coord_ti in sorted_coords:

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

    def plot_flow_field_Zplane(self, percent_height=0.5, show=True):
        zplane = int(self.x.shape[2] * percent_height)
        
        # plot flow field
        self.viz_manager.plot_constant_z(
            self.x[:, :, zplane], self.y[:, :, zplane], self.u_field[:, :, zplane])

        # plot turbines
        for coord, turbine in self.turbine_map.items():
            self.viz_manager.add_turbine_marker(turbine, coord, self.wind_direction)
        if show:
            self.viz_manager.show()

    def plot_flow_field_Xplane(self, percent_distance=0.4, show=True):
        xplane = int(self.x.shape[0] * percent_distance)

        # plot flow field
        self.viz_manager.plot_constant_x(
            self.y[:, xplane, :], self.z[:, xplane, :], self.u_field[:, xplane, :])

    def plot_flow_field_planes(self, heights=[0.5]):
        for height in heights:
            self.plot_flow_field_plane(height, False)
        self.viz_manager.show()

    # TODO def update_flowfield():
