# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
from ..utilities import Vec3
from ..utilities import cosd, sind, tand
import scipy as sp
from scipy.interpolate import griddata


class FlowField():
    """
    Object containing flow field information.

    FlowField is at the core of the FLORIS package. This class handles 
    the domain creation and initialization and computes the flow field 
    based on the input wake model and turbine map. It also contains 
    helper functions for quick flow field visualization.

    Args:
        wind_shear: A float that is the wind shear coefficient.
        wind_veer: A float that is the amount of veer across the rotor.
        wake: A container class :py:class:`floris.simulation.wake` with 
            wake model information used to calculate the flow field.
        wake_combination: A container class 
            :py:class:`floris.simulation.wake_combination` with wake 
            combination information.
        turbine_map: A :py:obj:`floris.simulation.turbine_map` object 
            that holds turbine information. 
        wind_map: A :py:obj:`floris.simulation.wind_map` object that 
            holds atmospheric input.

    Returns:
        FlowField: An instantiated FlowField object.
    """

    def __init__(self,
                 wind_shear,
                 wind_veer,
                 air_density,
                 wake,
                 turbine_map,
                 wind_map):

        self.reinitialize_flow_field(
            wind_shear=wind_shear,
            wind_veer=wind_veer,
            air_density=air_density,
            wake=wake,
            turbine_map=turbine_map,
            wind_map=wind_map,
            with_resolution=wake.velocity_model.model_grid_resolution
        )

    def _discretize_turbine_domain(self):
        """
        Create grid points at each turbine
        """
        xt = [coord.x1 for coord in self.turbine_map.coords]
        rotor_points = int(
            np.sqrt(self.turbine_map.turbines[0].grid_point_count))
        x_grid = np.zeros((len(xt), rotor_points, rotor_points))
        y_grid = np.zeros((len(xt), rotor_points, rotor_points))
        z_grid = np.zeros((len(xt), rotor_points, rotor_points))

        for i, (coord, turbine) in enumerate(self.turbine_map.items):
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
                    x_grid[i, j, k] = xt[i]
                    y_grid[i, j, k] = yt[j]
                    z_grid[i, j, k] = zt[k]

                    xoffset = x_grid[i, j, k] - coord.x1
                    yoffset = y_grid[i, j, k] - coord.x2
                    x_grid[i, j, k] = xoffset * cosd(-1 * self.wind_map.turbine_wind_direction[i]) - \
                        yoffset * sind(-1 * self.wind_map.turbine_wind_direction[i]) + coord.x1

                    y_grid[i, j, k] = yoffset * cosd(-1 * self.wind_map.turbine_wind_direction[i]) + \
                        xoffset * sind(-1 * self.wind_map.turbine_wind_direction[i]) + coord.x2
    
        return x_grid, y_grid, z_grid

    def _discretize_gridded_domain(self, xmin, xmax, ymin, ymax, zmin, zmax, resolution):
        """
        Generate a structured grid for the entire flow field domain.
        resolution: Vec3

        PF: NOTE, PERHAPS A BETTER NAME IS SETUP_GRIDDED_DOMAIN
        """
        x = np.linspace(xmin, xmax, int(resolution.x1))
        y = np.linspace(ymin, ymax, int(resolution.x2))
        z = np.linspace(zmin, zmax, int(resolution.x3))
        return np.meshgrid(x, y, z, indexing="ij")

    def _compute_initialized_domain(self, with_resolution=None, points=None):
        """
        Establish the layout of grid points for the flow field domain and 
        calculate initial values at these points.

        Note this function is currently complex to understand, and could be recast, but
        for now it has 3 main uses

        1) Initializing a non-curl model (gauss, multizone), using a call to _discretize_turbine_domain
        2) Initializing a gridded curl model (using a call to _discretize_gridded_domain)
        3) Appending points to a non-curl model, this could either be for adding additional points to calculate
            for use in visualization, or else to enable calculation of additional points.  Note this assumes
            the flow has previously been discritized in a prior call to _compute_initialized_domain / 
            _discretize_turbine_domain

        Args:
            points: An array that contains the x, y, and z coordinates of 
                user-specified points, at which the flow field velocity 
                is recorded. 
            with_resolution: Vec3

        Returns:
            *None* -- The flow field is updated directly in the 
            :py:class:`floris.simulation.floris.flow_field` object.  
        """
        if with_resolution is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = self.domain_bounds
            self.x, self.y, self.z = self._discretize_gridded_domain(
                xmin, xmax, ymin, ymax, zmin, zmax, with_resolution)
        else:
            if points is not None:

                # # # Alayna's Original method*******************************
                # Append matrices of idential points equal to the number of turbine grid points
                # print('APPEND THE POINTS')
                # shape = ((len(self.x)) + len(points[0]), len(self.x[0,:]), len(self.x[0,0,:]))
                # elem_shape = np.shape(self.x[0])
                # # print(elem_shape)
                # # print(np.full(elem_shape, points[0][0]))
                # # quit()

                
                # for i in range(len(points[0])):
                #     self.x = np.append(self.x, np.full(elem_shape, points[0][i]))
                #     self.y = np.append(self.y, np.full(elem_shape, points[1][i]))
                #     self.z = np.append(self.z, np.full(elem_shape, points[2][i]))
                # self.x = np.reshape(self.x, shape)
                # self.y = np.reshape(self.y, shape)
                # self.z = np.reshape(self.z, shape)
                # print('DONE APPEND THE POINTS')
                # # # END Alayna's Original method*******************************

                # # # Faster equivalent method I think*****************************
                # # Reshape same as above but vectorized I think
                # print('APPEND THE POINTS')
                # elem_num_el = np.size(self.x[0])
                # shape = ((len(self.x)) + len(points[0]), len(self.x[0,:]), len(self.x[0,0,:]))
                
                # self.x = np.append(self.x, np.repeat(points[0,:],elem_num_el))
                # self.y = np.append(self.y, np.repeat(points[1,:],elem_num_el))
                # self.z = np.append(self.z, np.repeat(points[2,:],elem_num_el))
                # self.x = np.reshape(self.x, shape)
                # self.y = np.reshape(self.y, shape)
                # self.z = np.reshape(self.z, shape)
                # print('DONE APPEND THE POINTS')
                # # # END Faster equivalent method I think*****************************

                # # Faster equivalent method with less final points ********************
                # Don't replicate points to be 25x (num points on turbine plane)
                # This will yield less overall points by removing redundant points and make later steps faster
                elem_num_el = np.size(self.x[0])
                num_points_to_add = len(points[0])
                matrices_to_add = int(np.ceil(num_points_to_add/elem_num_el))
                buffer_amount = matrices_to_add * elem_num_el - num_points_to_add
                shape = ((len(self.x)) + matrices_to_add, len(self.x[0,:]), len(self.x[0,0,:]))
                
                self.x = np.append(self.x, np.append(points[0,:],np.repeat(points[0,0],buffer_amount)))
                self.y = np.append(self.y, np.append(points[1,:],np.repeat(points[1,0],buffer_amount)))
                self.z = np.append(self.z, np.append(points[2,:],np.repeat(points[2,0],buffer_amount)))
                self.x = np.reshape(self.x, shape)
                self.y = np.reshape(self.y, shape)
                self.z = np.reshape(self.z, shape)
                # # Faster equivalent method with less final points ********************

            else:
                self.x, self.y, self.z = self._discretize_turbine_domain()


        # set grid point locations in wind_map
        self.wind_map.grid_layout = (self.x, self.y)

        # interpolate for initial values of flow field grid
        self.wind_map.calculate_turbulence_intensity(grid=True)
        self.wind_map.calculate_wind_direction(grid=True)
        self.wind_map.calculate_wind_speed(grid=True)

        self.u_initial = self.wind_map.grid_wind_speed * \
            (self.z / self.specified_wind_height)**self.wind_shear
        self.v_initial = np.zeros(np.shape(self.u_initial))
        self.w_initial = np.zeros(np.shape(self.u_initial))

        self.u = self.u_initial.copy()
        self.v = self.v_initial.copy()
        self.w = self.w_initial.copy()

    def _compute_turbine_velocity_deficit(self, x, y, z, turbine, coord, deflection, flow_field):
        return self.wake.velocity_function(x, y, z, turbine, coord, deflection, flow_field)

    def _compute_turbine_wake_deflection(self, x, y, z, turbine, coord, flow_field):
        return self.wake.deflection_function(x, y, z, turbine, coord, flow_field)

    def _rotated_grid(self, angle, center_of_rotation):
        """
        Rotate the discrete flow field grid.
        """
        xoffset = self.x - center_of_rotation.x1
        yoffset = self.y - center_of_rotation.x2
        rotated_x = xoffset * \
            cosd(angle) - yoffset * \
            sind(angle) + center_of_rotation.x1
        rotated_y = xoffset * \
            sind(angle) + yoffset * \
            cosd(angle) + center_of_rotation.x2
        return rotated_x, rotated_y, self.z

    def _rotated_dir(self, angle, center_of_rotation, rotated_map):
        """
        Rotate the discrete flow field grid and turbine map.
        """
        # get new boundaries for the wind farm once rotated
        x_coord = []
        y_coord = []
        for coord in rotated_map.coords:
            x_coord.append(coord.x1)
            y_coord.append(coord.x2)

        if self.wake.velocity_model.model_string == 'curl':
            # re-setup the grid for the curl model
            xmin = np.min(x_coord) - 2 * self.max_diameter
            xmax = np.max(x_coord) + 10 * self.max_diameter
            ymin = np.min(y_coord) - 2 * self.max_diameter
            ymax = np.max(y_coord) + 2 * self.max_diameter
            zmin = 0.1
            zmax = 6 * self.specified_wind_height

            # Save these bounds
            self._xmin = xmin
            self._xmax = xmax
            self._ymin = ymin
            self._ymax = ymax
            self._zmin = zmin
            self._zmax = zmax

            resolution = self.wake.velocity_model.model_grid_resolution
            self.x, self.y, self.z = self._discretize_gridded_domain(
                xmin, xmax, ymin, ymax, zmin, zmax, resolution)
            rotated_x, rotated_y, rotated_z = self._rotated_grid(
                0.0, center_of_rotation)
        else:
            rotated_x, rotated_y, rotated_z = self._rotated_grid(
                self.wind_map.grid_wind_direction, center_of_rotation)

        return rotated_x, rotated_y, rotated_z

    def _calculate_area_overlap(self, wake_velocities, freestream_velocities, turbine):
        """
        compute wake overlap based on the number of points that are not freestream velocity, i.e. affected by the wake
        """
        count = np.sum(freestream_velocities - wake_velocities <= 0.05)
        return (turbine.grid_point_count - count) / turbine.grid_point_count

    # Public methods

    def set_bounds(self, bounds_to_set=None):
        """
        A method that will set the domain bounds for the wake model.

        This method allows a user to customzie the domain bounds for 
        the current wake model being used, unless the wake model is the 
        Curl model, then a predefined domain is specified and used. If 
        the bounds are not specified, then a pre-defined set of bounds 
        will be used based on the mean wind direction at the turbines. 
        The bounds consist of the minimum and maximum values
        in the x-, y-, and z-directions.

        Args:
            bounds_to_set: A list of values representing the mininum 
                and maximum values for the domain 
                [xmin, xmax, ymin, ymax, zmin, zmax] 
                (default is *None*).

        Returns:
            *None* -- The flow field is updated directly in the 
            :py:class:`floris.simulation.floris.flow_field` object.
        """

        if self.wake.velocity_model.model_string == 'curl':
            # For the curl model, bounds are hard coded
            coords = self.turbine_map.coords
            x = [coord.x1 for coord in coords]
            y = [coord.x2 for coord in coords]
            eps = 0.1
            self._xmin = min(x) - 2 * self.max_diameter
            self._xmax = max(x) + 10 * self.max_diameter
            self._ymin = min(y) - 2 * self.max_diameter
            self._ymax = max(y) + 2 * self.max_diameter
            self._zmin = 0 + eps
            self._zmax = 6 * self.specified_wind_height

        elif bounds_to_set is not None:
            # Set the boundaries
            self._xmin = bounds_to_set[0]
            self._xmax = bounds_to_set[1]
            self._ymin = bounds_to_set[2]
            self._ymax = bounds_to_set[3]
            self._zmin = bounds_to_set[4]
            self._zmax = bounds_to_set[5]

        else:
            # Else, if none provided, use a shorter boundary for other models
            coords = self.turbine_map.coords
            x = [coord.x1 for coord in coords]
            y = [coord.x2 for coord in coords]
            eps = 0.1

            # find circular mean of wind directions at turbines
            wd = sp.stats.circmean(np.array(self.wind_map.turbine_wind_direction) * np.pi/180) * 180/np.pi

            # set bounds based on the mean wind direction to avoid
            # cutting off wakes near boundaries in visualization
            if wd < 270 and wd > 90: 
                self._xmin = min(x) - 10 * self.max_diameter
            else:
                self._xmin = min(x) - 2 * self.max_diameter

            if wd <= 90 or wd >= 270: 
                self._xmax = max(x) + 10 * self.max_diameter
            else:
                self._xmax = max(x) + 2 * self.max_diameter

            if wd <= 175 and wd >= 5: 
                self._ymin = min(y) - 5 * self.max_diameter
            else:
                self._ymin = min(y) - 2 * self.max_diameter

            if wd >= 185 and wd <= 355: 
                self._ymax = max(y) + 5 * self.max_diameter
            else:
                self._ymax = max(y) + 2 * self.max_diameter

            self._zmin = 0 + eps
            self._zmax = 2 * self.specified_wind_height

    def reinitialize_flow_field(self,
                                wind_shear=None,
                                wind_veer=None,
                                air_density=None,
                                wake=None,
                                turbine_map=None,
                                wind_map=None,
                                with_resolution=None,
                                bounds_to_set=None):
        """
        Reiniaitilzies the flow field when a parameter needs to be 
        updated.

        This method allows for changing/updating a variety of flow 
        related parameters. This would typically be used in loops or 
        optimizations where the user is calculating AEP over a wind 
        rose or investigating wind farm performance at different 
        conditions.

        Args:
            wind_shear: A float that is the wind shear coefficient 
                (default is *None*).
            wind_veer: A float that is the amount of veer across the 
                rotor (default is *None*).
            air_density: A float that is the air density (default is 
                *None*).
            wake: A container class :py:class:`floris.simulation.wake` 
                with wake model information used to calculate the flow 
                field (default is *None*).
            turbine_map: A :py:obj:`floris.simulation.turbine_map` 
                object that holds turbine information (default is 
                *None*).
            with_resolution: A :py:class:`floris.utilities.Vec3` object 
                that defines the flow field resolution at which to 
                calculate the wake (default is *None*).
            wind_map: A :py:obj:`floris.simulation.wind_map` object that 
                holds atmospheric input information.
            bounds_to_set: A list of values representing the mininum 
                and maximum values for the domain 
                [xmin, xmax, ymin, ymax, zmin, zmax] 
                (default is *None*).

        Returns:
            *None* -- The flow field is updated directly in the 
            :py:class:`floris.simulation.floris` object.
        """
        # reset the given parameters
        if turbine_map is not None:
            self.turbine_map = turbine_map
        if wind_map is not None:
            self.wind_map = wind_map           
        if wind_shear is not None:
            self.wind_shear = wind_shear
        if wind_veer is not None:
            self.wind_veer = wind_veer
        if air_density is not None:
            self.air_density = air_density
            for turbine in self.turbine_map.turbines:
                turbine.air_density = self.air_density
        if wake is not None:
            self.wake = wake
        if with_resolution is None:
            with_resolution = self.wake.velocity_model.model_grid_resolution

        # initialize derived attributes and constants
        self.max_diameter = max(
            [turbine.rotor_diameter for turbine in self.turbine_map.turbines])
        self.specified_wind_height = self.turbine_map.turbines[0].hub_height

        # Set the domain bounds
        self.set_bounds(bounds_to_set=bounds_to_set)

        # reinitialize the flow field
        self._compute_initialized_domain(with_resolution=with_resolution)

        # reinitialize the turbines
        for i, turbine in enumerate(self.turbine_map.turbines):
            turbine.reinitialize_turbine(self.wind_map.turbine_turbulence_intensity[i])

    def calculate_wake(self, no_wake=False, points=None):
        """
        Updates the flow field based on turbine activity.

        This method rotates the turbine farm such that the wind 
        direction is coming from 270 degrees. It then loops over the 
        turbines, updating their velocities, calculating the wake 
        deflection/deficit, and combines the wake with the flow field.

        Args:
            no_wake: A bool that when *True* updates the turbine 
                quantities without calculating the wake or adding the 
                wake to the flow field.
            points: An array that contains the x, y, and z coordinates of 
                user-specified points at which the flow field velocity 
                is recorded. 

        Returns:
            *None* -- The flow field and turbine properties are updated 
            directly in the :py:class:`floris.simulation.floris` object.
        """

        if points is not None:
            # add points to flow field grid points
            self._compute_initialized_domain(points=points)

        # reinitialize the turbines
        for i, turbine in enumerate(self.turbine_map.turbines):
            turbine.reinitialize_turbine(self.wind_map.turbine_turbulence_intensity[i])

        # define the center of rotation with reference to 270 deg as center of flow field
        x0 = np.mean([np.min(self.x) , np.max(self.x)])
        y0 = np.mean([np.min(self.y) , np.max(self.y)])
        center_of_rotation = Vec3(x0, y0, 0)
        
        # Rotate the turbines such that they are now in the frame of reference
        # of the wind direction simpifying computing the wakes and wake overlap
        rotated_map = self.turbine_map.rotated(
            self.wind_map.turbine_wind_direction, center_of_rotation)

        # rotate the discrete grid and turbine map
        initial_rotated_x, initial_rotated_y, rotated_z = self._rotated_dir(
                self.wind_map.grid_wind_direction, center_of_rotation, rotated_map)

        # sort the turbine map
        sorted_map = rotated_map.sorted_in_x_as_list()

        # calculate the velocity deficit and wake deflection on the mesh
        u_wake = np.zeros(np.shape(self.u))
        v_wake = np.zeros(np.shape(self.u))
        w_wake = np.zeros(np.shape(self.u))
        
        rx = np.zeros(len(self.turbine_map.coords))
        ry = np.zeros(len(self.turbine_map.coords))
        for i, cord in enumerate(self.turbine_map.coords):
            rx[i], ry[i] = cord.x1prime, cord.x2prime   

        for coord, turbine in sorted_map:
            xloc,yloc = np.array(rx == coord.x1),np.array(ry == coord.x2)
            idx = int(np.where(np.logical_and(yloc == True, xloc == True))[0])

            if np.unique(self.wind_map.grid_wind_direction).size == 1:
                # only rotate grid once for homogeneous wind direction
                rotated_x, rotated_y = initial_rotated_x, initial_rotated_y
        
            else:   
                # adjust grid rotation with respect to current turbine for heterogeneous wind direction
                wd = self.wind_map.turbine_wind_direction[idx] - self.wind_map.grid_wind_direction

                xoffset = center_of_rotation.x1 - rx[idx] # for straight wakes, change rx[idx] to initial_rotated_x
                yoffset = center_of_rotation.x2 - ry[idx] # for straight wakes, change ry[idx] to initial_rotated_y
                y_grid_offset = xoffset * sind(wd) + yoffset * cosd(wd) - yoffset
                rotated_y = initial_rotated_y - y_grid_offset

                xoffset = center_of_rotation.x1 - initial_rotated_x
                yoffset = center_of_rotation.x2 - initial_rotated_y
                x_grid_offset = xoffset * cosd(wd) - yoffset * sind(wd) - xoffset
                rotated_x = initial_rotated_x - x_grid_offset 

            # update the turbine based on the velocity at its hub
            turbine.update_velocities(
                u_wake, coord, self, rotated_x, rotated_y, rotated_z)

            # get the wake deflecton field
            deflection = self._compute_turbine_wake_deflection(
                rotated_x, rotated_y, rotated_z, turbine, coord, self)   

            # get the velocity deficit accounting for the deflection
            turb_u_wake, turb_v_wake, turb_w_wake = self._compute_turbine_velocity_deficit(
                rotated_x, rotated_y, rotated_z, turbine, coord, deflection, self)

            # include turbulence model for the gaussian wake model from Porte-Agel
            if self.wake.velocity_model.model_string == 'gauss' or \
                self.wake.velocity_model.model_string == 'gauss_curl_hybrid':

                # compute area overlap of wake on other turbines and update downstream turbine turbulence intensities
                for coord_ti, turbine_ti in sorted_map:
                    xloc,yloc = np.array(rx == coord_ti.x1),np.array(ry == coord_ti.x2)
                    idx = int(np.where(np.logical_and(yloc == True, xloc == True))[0])
                    
                    if coord_ti.x1 > coord.x1 and np.abs(coord.x2 - coord_ti.x2) < 2*turbine.rotor_diameter:  
                        # only assess the effects of the current wake

                        freestream_velocities = turbine_ti.calculate_swept_area_velocities(
                            self.u_initial,
                            coord_ti,
                            rotated_x,
                            rotated_y,
                            rotated_z)

                        wake_velocities = turbine_ti.calculate_swept_area_velocities(
                            self.u_initial - turb_u_wake,
                            coord_ti,
                            rotated_x,
                            rotated_y,
                            rotated_z)

                        area_overlap = self._calculate_area_overlap(
                            wake_velocities, freestream_velocities, turbine)

                        if area_overlap > 0.0 and coord_ti.x1 <= 15*turbine.rotor_diameter + coord.x1:
                            turbine_ti.current_turbulence_intensity = turbine_ti.calculate_turbulence_intensity(
                                area_overlap,
                                self.wind_map.turbine_turbulence_intensity[idx],
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

        # rotate the grid if it is curl
        if self.wake.velocity_model.model_string == 'curl':
            self.x, self.y, self.z = self._rotated_grid(
                -1 * self.wind_map.grid_wind_direction, center_of_rotation)

    # Getters & Setters
    @property
    def domain_bounds(self):
        """
        Property that returns the bounds of the flow field domain.

        Returns:
            floats: xmin, xmax, ymin, ymax, zmin, zmax

            The mininmum and maxmimum values of the domain in the x, y, and z directions.

        Examples:
            To get the domain bounds:

            >>> xmin, xmax, ymin, ymax, zmin, zmax = 
            ... floris.farm.flow_field.domain_bounds()
        """
        return self._xmin, self._xmax, self._ymin, self._ymax, self._zmin, self._zmax
