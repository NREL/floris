# Copyright 2017 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
from .types import Vec3
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
                 turbine_map):

        self.wind_speed = wind_speed
        self._wind_direction = wind_direction
        self.wind_shear = wind_shear
        self.wind_veer = wind_veer
        self.turbulence_intensity = turbulence_intensity
        self.air_density = air_density
        self.wake = wake
        self.turbine_map = turbine_map
        
        # initialize derived attributes and constants
        self.max_diameter = max([turbine.rotor_diameter for turbine in self.turbine_map.turbines])
        self.specified_wind_height = self.turbine_map.turbines[0].hub_height
        self.reinitialize_flow_field(with_resolution=self.wake.velocity_model.model_grid_resolution)

    def _discretize_turbine_domain(self):
        """
        Create grid points at each turbine
        """
        xt = [coord.x1 for coord in self.turbine_map.coords]
        rotor_points = int(np.sqrt(self.turbine_map.turbines[0].grid_point_count))
        x_grid = np.zeros((len(xt), rotor_points, rotor_points))
        y_grid = np.zeros((len(xt), rotor_points, rotor_points))
        z_grid = np.zeros((len(xt), rotor_points, rotor_points))

        for i, (coord, turbine) in enumerate(self.turbine_map.items()):
            xt = [coord.x1 for coord in self.turbine_map.coords]
            yt = np.linspace(
                coord.x2 - turbine.rotor_radius,
                coord.x2 + turbine.rotor_radius,
                rotor_points
            )
            zt = np.linspace(
                coord.x3 - turbine.rotor_radius,
                coord.x3 + turbine.rotor_radius,
                rotor_points
            )

            for j in range(len(yt)):
                for k in range(len(zt)):
                    x_grid[i,j,k] = xt[i]
                    y_grid[i,j,k] = yt[j]
                    z_grid[i,j,k] = zt[k]

                    xoffset = x_grid[i,j,k] - coord.x1
                    yoffset = y_grid[i,j,k] - coord.x2
                    x_grid[i,j,k] = xoffset * np.cos(-1 * self.wind_direction) - yoffset * np.sin(-1 * self.wind_direction) + coord.x1
                    y_grid[i,j,k] = yoffset * np.cos(-1 * self.wind_direction) + xoffset * np.sin(-1 * self.wind_direction) + coord.x2
        
        return x_grid, y_grid, z_grid

    def _discretize_freestream_domain(self, xmin, xmax, ymin, ymax, zmin, zmax, resolution):
        """
        Generate a structured grid for the entire flow field domain.
        resolution: Vec3
        """
        x = np.linspace(xmin, xmax, int(resolution.x1))
        y = np.linspace(ymin, ymax, int(resolution.x2))
        z = np.linspace(zmin, zmax, int(resolution.x3))
        return np.meshgrid(x, y, z, indexing="ij")

    def _compute_initialized_domain(self, with_resolution=None):
        if with_resolution is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = self._get_domain_bounds()
            self.x, self.y, self.z = self._discretize_freestream_domain(xmin, xmax, ymin, ymax, zmin, zmax, with_resolution)
        else:
            self.x, self.y, self.z = self._discretize_turbine_domain()

        self.u_initial = self.wind_speed * (self.z / self.specified_wind_height)**self.wind_shear
        self.v_initial = np.zeros(np.shape(self.u_initial))
        self.w_initial = np.zeros(np.shape(self.u_initial))

        self.u = self.u_initial.copy()
        self.v = self.v_initial.copy()
        self.w = self.w_initial.copy()

    def _get_domain_bounds(self):
        coords = self.turbine_map.coords
        x = [coord.x1 for coord in coords]
        y = [coord.x2 for coord in coords]
        eps = 0.1
        xmin = min(x) - 2 * self.max_diameter
        xmax = max(x) + 10 * self.max_diameter
        ymin = min(y) - 2 * self.max_diameter
        ymax = max(y) + 2 * self.max_diameter
        zmin = 0 + eps 
        zmax = 6 * self.specified_wind_height
        return xmin, xmax, ymin, ymax, zmin, zmax

    def _compute_turbine_velocity_deficit(self, x, y, z, turbine, coord, deflection, wake, flow_field):
        return self.wake.velocity_function(x, y, z, turbine, coord, deflection, wake, flow_field)

    def _compute_turbine_wake_deflection(self, x, y, turbine, coord, flow_field):
        return self.wake.deflection_function(x, y, turbine, coord, flow_field)

    def _rotated_grid(self, angle, center_of_rotation):
        xoffset = self.x - center_of_rotation.x1
        yoffset = self.y - center_of_rotation.x2
        rotated_x = xoffset * \
            np.cos(angle) - yoffset * \
            np.sin(angle) + center_of_rotation.x1
        rotated_y = xoffset * \
            np.sin(angle) + yoffset * \
            np.cos(angle) + center_of_rotation.x2
        return rotated_x, rotated_y, self.z

    def _calculate_area_overlap(self, wake_velocities, freestream_velocities, turbine):
        """
        compute wake overlap based on the number of points that are not freestream velocity, i.e. affected by the wake
        """
        count = np.sum(freestream_velocities - wake_velocities <= 0.05)
        return (turbine.grid_point_count - count) / turbine.grid_point_count

    # Public methods    
    def reinitialize_flow_field(self,
                                wind_speed=None,
                                wind_direction=None,
                                wind_shear=None,
                                turbulence_intensity=None,
                                with_resolution=None):
        # reset the given parameters
        if wind_speed is not None:
            self.wind_speed = wind_speed
        if wind_direction is not None:
            self.wind_direction = wind_direction
        if wind_shear is not None:
            self.wind_shear = wind_shear
        if turbulence_intensity is not None:
            self.turbulence_intensity = turbulence_intensity

        # reinitialize the flow field
        self._compute_initialized_domain(with_resolution=with_resolution)

    def calculate_wake(self, no_wake=False):

        # initialize turbulence intensity at every turbine (seems sloppy)
        for coord, turbine in self.turbine_map.items():
            turbine.air_density = self.air_density

        # rotate the discrete grid and turbine map
        center_of_rotation = Vec3(0, 0, 0)
        rotated_x, rotated_y, rotated_z = self._rotated_grid(self.wind_direction, center_of_rotation)

        # Rotate the turbines such that they are now in the frame of reference 
        # of the wind direction simpifying computing the wakes and wake overlap
        rotated_map = self.turbine_map.rotated(self.wind_direction, center_of_rotation)

        # sort the turbine map
        sorted_map = rotated_map.sorted_in_x_as_list()

        # calculate the velocity deficit and wake deflection on the mesh
        u_wake = np.zeros(np.shape(self.u))
        v_wake = np.zeros(np.shape(self.u))
        w_wake = np.zeros(np.shape(self.u))
        for coord, turbine in sorted_map:

            # update the turbine based on the velocity at its hub
            turbine.update_velocities(u_wake, coord, self, rotated_x, rotated_y, rotated_z)
            
            # get the wake deflecton field
            deflection = self._compute_turbine_wake_deflection(rotated_x, rotated_y, turbine, coord, self)

            # get the velocity deficit accounting for the deflection
            if self.wake.velocity_model.requires_resolution:
                turb_u_wake, turb_v_wake, turb_w_wake = self._compute_turbine_velocity_deficit(rotated_x, rotated_y, rotated_z, turbine, coord, deflection, self.wake, self)
            else:
                turb_u_wake = self._compute_turbine_velocity_deficit(rotated_x, rotated_y, rotated_z, turbine, coord, deflection, self.wake, self)
                turb_v_wake = np.zeros(self.u.shape)
                turb_w_wake = np.zeros(self.u.shape)

            if self.wake.velocity_model.requires_resolution:

                # compute area overlap of wake on other turbines and update downstream turbine turbulence intensities
                for coord_ti, turbine_ti in sorted_map:

                    if coord_ti.x1 > coord.x1 and np.abs(coord.x2 - coord_ti.x2) < 2*turbine.rotor_diameter:
                        # only assess the effects of the current wake
                        
                        freestream_velocities = turbine_ti._calculate_swept_area_velocities(
                            self.wind_direction,
                            self.u_initial,
                            coord_ti,
                            rotated_x,
                            rotated_y,
                            rotated_z)

                        wake_velocities = turbine_ti._calculate_swept_area_velocities(
                            self.wind_direction,
                            self.u_initial - turb_u_wake,
                            coord_ti,
                            rotated_x,
                            rotated_y,
                            rotated_z)

                        area_overlap = self._calculate_area_overlap(wake_velocities, freestream_velocities, turbine)
                        if area_overlap > 0.0:
                            turbine_ti.turbulence_intensity = turbine_ti.calculate_turbulence_intensity(
                                self.turbulence_intensity,
                                self.wake.velocity_model,
                                coord_ti,
                                coord,
                                turbine
                            )

            # combine this turbine's wake into the full wake field
            if not no_wake:
                # TODO: why not use the wake combination scheme in every component?
                u_wake = self.wake.combination_function(u_wake, turb_u_wake)
                v_wake = (v_wake + turb_v_wake)
                w_wake = (w_wake + turb_w_wake)

        # apply the velocity deficit field to the freestream
        if not no_wake:
            # TODO: are these signs correct?
            self.u = self.u_initial - u_wake
            self.v = self.v_initial + v_wake
            self.w = self.w_initial + w_wake

    def get_flow_field_with_resolution(self, resolution):
        """
        resolution: Vec3()
        """
        if self.wake.velocity_model.requires_resolution and \
            self.wake.velocity_model.requires_resolution != resolution:
            print("WARNING: The current wake velocity model contains a required grid resolution;")
            print("    The Resolution given to FlowField.get_flow_field_with_resolution is ignored.")
            resolution = self.wake.velocity_model.grid_resolution
        xmin, xmax, ymin, ymax, zmin, zmax = self._get_domain_bounds()
        self.x, self.y, self.z = self._discretize_freestream_domain(xmin, xmax, ymin, ymax, zmin, zmax, resolution)
        self.reinitialize_flow_field(with_resolution=resolution)
        return self.u, self.v, self.w

    # Getters & Setters
    @property
    def wind_direction(self):
        return self._wind_direction
    
    @wind_direction.setter
    def wind_direction(self, value):
        # frame of reference is west
        self._wind_direction = np.radians(value - 270)
