# Copyright 2017 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
from .coordinate import Coordinate
from scipy.interpolate import griddata

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
                 air_density,
                 wake,
                 wake_combination,
                 turbine_map):

        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.wind_shear = wind_shear
        self.wind_veer = wind_veer
        self.turbulence_intensity = turbulence_intensity
        self.air_density = air_density
        self.wake = wake
        self.wake_combination = wake_combination
        self.turbine_map = turbine_map
        
        # initialize derived attributes and constants
        self.max_diameter = max(
            [turbine.rotor_diameter for turbine in self.turbine_map.turbines])
        self.hub_height = self.turbine_map.turbines[0].hub_height
        self.x, self.y, self.z = self._discretize_turbine_domain()
        self.initial_flowfield = self._initial_flowfield()
        self.u_field = self._initial_flowfield()

    def _discretize_turbine_domain(self):
        """
        Create grid points at each turbine
        """
        xt = [coord.x for coord in self.turbine_map.coords]
        rotor_points = int(np.sqrt(self.turbine_map.turbines[0].grid_point_count))
        x_grid = np.zeros((len(xt), rotor_points, rotor_points))
        y_grid = np.zeros((len(xt), rotor_points, rotor_points))
        z_grid = np.zeros((len(xt), rotor_points, rotor_points))

        for i, (coord, turbine) in enumerate(self.turbine_map.items()):
            yt = np.linspace(coord.y - turbine.rotor_radius,
                             coord.y + turbine.rotor_radius,
                             rotor_points)
            zt = np.linspace(turbine.hub_height - turbine.rotor_radius,
                             turbine.hub_height + turbine.rotor_radius,
                             rotor_points)

            for j in range(len(yt)):
                for k in range(len(zt)):
                    x_grid[i,j,k] = xt[i]
                    y_grid[i,j,k] = yt[j]
                    z_grid[i,j,k] = zt[k]

                    xoffset = x_grid[i,j,k] - coord.x
                    yoffset = y_grid[i,j,k] - coord.y
                    x_grid[i,j,k] = xoffset * np.cos(-self.wind_direction) - yoffset * np.sin(-self.wind_direction) + coord.x
                    y_grid[i,j,k] = yoffset * np.cos(-self.wind_direction) + xoffset * np.sin(-self.wind_direction) + coord.y
        
        return x_grid, y_grid, z_grid
    
    def _initial_flowfield(self):
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
            turbine.turbulence_intensity = self.turbulence_intensity
            turbine.air_density = self.air_density

        # rotate the discrete grid and turbine map
        center_of_rotation = Coordinate(0,0)

        rotated_x, rotated_y, rotated_z = self._rotated_grid(
            self.wind_direction, center_of_rotation)

        # Rotate the turbines such that they are now in the frame of reference 
        # of the wind direction simpifying computing the wakes and wake overlap
        rotated_map = self.turbine_map.rotated(
            self.wind_direction, center_of_rotation)

        # sort the turbine map
        sorted_map = rotated_map.sorted_in_x_as_list()

        # calculate the velocity deficit and wake deflection on the mesh
        u_wake = np.zeros(self.u_field.shape)
        for coord, turbine in sorted_map:

            # update the turbine based on the velocity at its hub
            turbine.update_quantities(u_wake, coord, self, rotated_x, rotated_y, rotated_z)
            
            # get the wake deflecton field
            deflection = self._compute_turbine_wake_deflection(rotated_x, rotated_y, turbine, coord, self)

            # get the velocity deficit accounting for the deflection
            turb_wake = self._compute_turbine_velocity_deficit(
                rotated_x, rotated_y, rotated_z, turbine, coord, deflection, self.wake, self)

            if self.wake.velocity_model.type_string == 'gauss':

                # compute area overlap of wake on other turbines and update downstream turbine turbulence intensities
                for coord_ti, turbine_ti in sorted_map:

                    if coord_ti.x > coord.x and np.abs(coord.y - coord_ti.y) < 2*turbine.rotor_diameter:
                        # only assess the effects of the current wake

                        if turbine_ti.plotting:
                            wake_velocities = turbine_ti._calculate_swept_area_velocities_visualization(
                                self.grid_resolution,
                                self.initial_flowfield - turb_wake,
                                coord_ti,
                                rotated_x,
                                rotated_y,
                                rotated_z)
                            freestream_velocities = turbine_ti._calculate_swept_area_velocities_visualization(
                                self.grid_resolution,
                                self.initial_flowfield,
                                coord_ti,
                                rotated_x,
                                rotated_y,
                                rotated_z)

                        else:
                            wake_velocities = turbine_ti._calculate_swept_area_velocities(
                                self.wind_direction,
                                self.initial_flowfield - turb_wake,
                                coord_ti,
                                rotated_x,
                                rotated_y,
                                rotated_z)
                            freestream_velocities = turbine_ti._calculate_swept_area_velocities(
                                self.wind_direction,
                                self.initial_flowfield,
                                coord_ti,
                                rotated_x,
                                rotated_y,
                                rotated_z)

                        area_overlap = self._calculate_area_overlap(wake_velocities, freestream_velocities, turbine)
                        if area_overlap > 0.0:
                            turbine_ti.turbulence_intensity = turbine_ti.calculate_turbulence_intensity(
                                                self.turbulence_intensity,
                                                self.wake.velocity_model, coord_ti, coord, turbine)

            # combine this turbine's wake into the full wake field
            u_wake = self.wake_combination.combine(u_wake, turb_wake)

        # apply the velocity deficit field to the freestream
        self.u_field = self.initial_flowfield - u_wake
