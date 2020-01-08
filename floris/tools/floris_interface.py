# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
import pandas as pd
from floris.simulation import Floris
from floris.simulation import TurbineMap
from .flow_data import FlowData
from ..utilities import Vec3
import copy
from scipy.stats import norm
from floris.simulation import WindMap
from .cut_plane import CutPlane, get_plane_from_flow_data


class FlorisInterface():
    """
    The interface between a FLORIS instance and the wfc tools
    """

    def __init__(self, input_file=None, input_dict=None):
        if input_file is None and input_dict is None:
            raise ValueError('Input file or dictionary must be supplied')
        self.input_file = input_file
        self.floris = Floris(input_file=input_file, input_dict=input_dict)

    def calculate_wake(self, yaw_angles=None, no_wake=False, points = None):
        """
        Wrapper to the floris flow field calculate_wake method

        Args:
            yaw_angles (np.array, optional): Turbine yaw angles.
                Defaults to None.
            no_wake: (bool, optional): When *True* updates the turbine
                quantities without calculating the wake or adding the
                wake to the flow field. Defaults to False.
            points: (np.array, optional): The x, y, and z coordinates at 
                which the flow field velocity is to be recorded. Defaults
                to None.
        """

        if yaw_angles is not None:
            self.floris.farm.set_yaw_angles(yaw_angles)

        self.floris.farm.flow_field.calculate_wake(no_wake=no_wake, points=points)

    def reinitialize_flow_field(self,
                                wind_speed=None,
                                wind_layout=None,
                                wind_direction=None,
                                wind_shear=None,
                                wind_veer=None,
                                turbulence_intensity=None,
                                turbulence_kinetic_energy=None,
                                air_density=None,
                                wake=None,
                                layout_array=None,
                                with_resolution=None):
        """
        Wrapper to
        :py:meth:`floris.simlulation.flow_field.reinitialize_flow_field`.
        All input values are used to update the flow_field instance.

        Args:
            wind_speed (list, optional): background wind speed.
                Defaults to None.
            wind_layout (tuple, optional): tuple of x- and
                y-locations of wind speed measurements. 
                Defaults to None.
            wind_direction (list, optional): background wind direction.
                Defaults to None.
            wind_shear (float, optional): shear exponent.
                Defaults to None.
            wind_veer (float, optional): direction change over rotor.
                Defaults to None.
            turbulence_intensity (list, optional): background turbulence 
                intensity. Defaults to None.
            turbulence_kinetic_energy (list, optional): background turbulence
                kinetic energy. Defaults to None.
            air_density (float, optional): ambient air density.
                Defaults to None.
            wake (:py:class:`floris.simulation.wake`, optional): A container
                class :py:class:`floris.simulation.wake` with wake model
                information used to calculate the flow field. Defaults to None.
            layout_array (np.array, optional): array of x- and
                y-locations of wind turbines. Defaults to None.
            with_resolution (float, optional): resolution of output
                flow_field. Defaults to None.
        """

        wind_map = self.floris.farm.wind_map
        turbine_map = self.floris.farm.flow_field.turbine_map
        if turbulence_kinetic_energy is not None:
            if wind_speed == None: wind_map.input_speed
            turbulence_intensity = self.TKE_to_TI(turbulence_kinetic_energy, wind_speed)

        if wind_layout or layout_array is not None:
            # Build turbine map and wind map (convenience layer for user)
            if layout_array is None: layout_array = (self.layout_x, self.layout_y)  
            else:
                turbine_map = TurbineMap(                
                    layout_array[0], layout_array[1], \
                    [copy.deepcopy(self.floris.farm.turbines[0]) \
                     for ii in range(len(layout_array[0]))])          
            if wind_layout is None: wind_layout = wind_map.wind_layout 
            if wind_speed is None: wind_speed = wind_map.input_speed
            if wind_direction is None: wind_direction = wind_map.input_direction
            if turbulence_intensity is None: turbulence_intensity = wind_map.input_ti 

            wind_map = WindMap(wind_speed=wind_speed,
                          layout_array=layout_array,
                          wind_layout=wind_layout, 
                          turbulence_intensity=turbulence_intensity, 
                          wind_direction=wind_direction
                          )
            self.floris.farm.wind_map = wind_map
                
        else:
            turbine_map = None

            if wind_speed is not None:
                wind_map.input_speed = wind_speed
                wind_map.calculate_wind_speed()
            
            if turbulence_intensity is not None:
                wind_map.input_ti = turbulence_intensity
                wind_map.calculate_turbulence_intensity()

            if wind_direction is not None:
                wind_map.input_direction = wind_direction
                wind_map.calculate_wind_direction()
           
            # redefine wind_map in Farm object
            self.floris.farm.wind_map = wind_map
            
        self.floris.farm.flow_field.reinitialize_flow_field(
            wind_shear=wind_shear,
            wind_veer=wind_veer,
            air_density=air_density,
            wake=wake,
            turbine_map=turbine_map,
            with_resolution=with_resolution,
            wind_map = self.floris.farm.wind_map)

    def get_plane_of_points(self,
                                  x1_resolution=200, 
                                  x2_resolution=200, 
                                  normal_vector='z',
                                  x3_value=100,
                                  x1_bounds=None,
                                  x2_bounds=None):

        """
        Use points method in calculate_wake to quick extract a slice of points

        Args:
            x1_resolution (float, optional): output array resolution.
                Defaults to 200.
            x2_resolution (float, optional): output array resolution.
                Defaults to 200.
            normal_vector (string, optional): vector normal to plane
                Defaults to z.
            x3_value (float, optional): value of normal vector to slice through
                Defaults to 100.
            x1_bounds (tuple, optional): limits of output array.
                Defaults to None.
            x2_bounds (tuple, optional): limits of output array.
                Defaults to None.

        Returns:
            dataframe of x1,x2,u,v,w values
        """

        # Get a copy for the flow field so don't change underlying grid points
        flow_field = copy.deepcopy(self.floris.farm.flow_field)

        if self.floris.farm.flow_field.wake.velocity_model.requires_resolution:
            
            # If this is a gridded model, must extract from full flow field
            print('Model identifed as %s requires use of underyling grid print' % self.floris.farm.flow_field.wake.velocity_model.model_string)

            # Get the flow data and extract the plane using it
            flow_data = self.get_flow_data()
            return get_plane_from_flow_data(flow_data,normal_vector=normal_vector, x3_value=x3_value)



        # If x1 and x2 bounds are not provided, use rules of thumb
        if normal_vector == 'z': # Rules of thumb for horizontal plane
            if x1_bounds is None:
                coords = self.floris.farm.flow_field.turbine_map.coords
                max_diameter = self.floris.farm.flow_field.max_diameter
                x = [coord.x1 for coord in coords]
                x1_bounds = (min(x) - 2 * max_diameter, max(x) + 10 * max_diameter)
            if x2_bounds is None:
                coords = self.floris.farm.flow_field.turbine_map.coords
                max_diameter = self.floris.farm.flow_field.max_diameter
                y = [coord.x2 for coord in coords]
                x2_bounds = (min(y) - 2 * max_diameter, max(y) + 2 * max_diameter)
        if normal_vector == 'x': # Rules of thumb for cut plane plane
            if x1_bounds is None:
                coords = self.floris.farm.flow_field.turbine_map.coords
                max_diameter = self.floris.farm.flow_field.max_diameter
                y = [coord.x2 for coord in coords]
                x1_bounds = (min(y) - 2 * max_diameter, max(y) + 2 * max_diameter)
            if x2_bounds is None:
                hub_height = self.floris.farm.flow_field.turbine_map.turbines[0].hub_height
                x2_bounds = (0,hub_height)


        # Set up the points to test
        x1_array = np.linspace(x1_bounds[0],x1_bounds[1],num=x1_resolution)
        x2_array = np.linspace(x2_bounds[0],x2_bounds[1],num=x2_resolution)

        # Grid the points and flatten
        x1_array, x2_array = np.meshgrid(x1_array, x2_array)
        x1_array = x1_array.flatten()
        x2_array = x2_array.flatten()
        x3_array = np.ones_like(x1_array) * x3_value

        # Create the points matrix
        if normal_vector == 'z':
            points = np.row_stack((x1_array,x2_array,x3_array))
        if normal_vector == 'x':
            points = np.row_stack((x3_array,x1_array,x2_array))

        # Recalcuate wake with these points
        flow_field.calculate_wake(points=points)

        # Get results vectors
        x_flat = flow_field.x.flatten()
        y_flat = flow_field.y.flatten()
        z_flat = flow_field.z.flatten()
        u_flat = flow_field.u.flatten()
        v_flat = flow_field.v.flatten()
        w_flat = flow_field.w.flatten()

        # Create a df of these
        if normal_vector == 'z':
            df = pd.DataFrame({'x1':x_flat,
            'x2':y_flat,
            'x3':z_flat,
            'u':u_flat,
            'v':v_flat,
            'w':w_flat
            })
        if normal_vector == 'x':
            df = pd.DataFrame({'x1':y_flat,
            'x2':z_flat,
            'x3':x_flat,
            'u':u_flat,
            'v':v_flat,
            'w':w_flat
            })
        if normal_vector == 'y':
            df = pd.DataFrame({'x1':x_flat,
            'x2':z_flat,
            'x3':y_flat,
            'u':u_flat,
            'v':v_flat,
            'w':w_flat
            })

        # Subset to plane
        df = df[df.x3==x3_value]

        # Drop duplicates
        df = df.drop_duplicates()

        # Limit to requested points
        df = df[df.x1.isin(x1_array)]
        df = df[df.x2.isin(x2_array)]

        # Return the dataframe
        return df

    def get_set_of_points(self,
                                  x_points,
                                  y_points,
                                  z_points):

        """
        Use points method in calculate_wake to quick extract a slice of points

        Args:
            x_points, float, array of floats
            y_points, float, array of floats
            z_points, float, array of floats


        Returns:
            dataframe of x,y,z,u,v,w values
        """

        # Get a copy for the flow field so don't change underlying grid points
        flow_field = copy.deepcopy(self.floris.farm.flow_field)

        if self.floris.farm.flow_field.wake.velocity_model.requires_resolution:
            
            # If this is a gridded model, must extract from full flow field
            print('Model identifed as %s requires use of underyling grid print' % self.floris.farm.flow_field.wake.velocity_model.model_string)
            print('FUNCTION NOT AVAILABLE CURRENTLY')

        # Set up points matrix
        points = np.row_stack((x_points,y_points,z_points))

        # Recalcuate wake with these points
        flow_field.calculate_wake(points=points)

        # Get results vectors
        x_flat = flow_field.x.flatten()
        y_flat = flow_field.y.flatten()
        z_flat = flow_field.z.flatten()
        u_flat = flow_field.u.flatten()
        v_flat = flow_field.v.flatten()
        w_flat = flow_field.w.flatten()

        df = pd.DataFrame({'x':x_flat,
            'y':y_flat,
            'z':z_flat,
            'u':u_flat,
            'v':v_flat,
            'w':w_flat
            })


        # Subset to points requests
        df = df[df.x.isin(x_points)]
        df = df[df.y.isin(y_points)]
        df = df[df.z.isin(z_points)]

        # Drop duplicates
        df = df.drop_duplicates()

        # Return the dataframe
        return df


    def get_hor_plane(self, height=None,
                x_resolution=200, 
                y_resolution=200, 
                x_bounds=None,
                y_bounds=None):
        """
        Get a horizontal cut through plane at a specific height

        Args:
            height (float): height of cut plane, defaults to hub-height
                Defaults to Hub-height.
            x1_resolution (float, optional): output array resolution.
                Defaults to 200.
            x2_resolution (float, optional): output array resolution.
                Defaults to 200.
            x1_bounds (tuple, optional): limits of output array.
                Defaults to None.
            x2_bounds (tuple, optional): limits of output array.
                Defaults to None.

        Returns:
            horplane
        """

        # If height not provided, use the hub height
        if height is None:
            height = self.floris.farm.flow_field.turbine_map.turbines[0].hub_height
            print('Default to hub height: %.1f' % height)

        # Get the points of data in a dataframe
        df = self.get_plane_of_points(
                                  x1_resolution=x_resolution, 
                                  x2_resolution=y_resolution, 
                                  normal_vector='z',
                                  x3_value=height,
                                  x1_bounds=x_bounds,
                                  x2_bounds=y_bounds)

        # Compute and return the cutplane
        return CutPlane(df)


    def get_cross_plane(self, x_loc,
                x_resolution=200, 
                y_resolution=200, 
                x_bounds=None,
                y_bounds=None):
        """
        Get a horizontal cut through plane at a specific height

        Args:
            height (float): height of cut plane, defaults to hub-height
                Defaults to Hub-height.
            x1_resolution (float, optional): output array resolution.
                Defaults to 200.
            x2_resolution (float, optional): output array resolution.
                Defaults to 200.
            x1_bounds (tuple, optional): limits of output array.
                Defaults to None.
            x2_bounds (tuple, optional): limits of output array.
                Defaults to None.

        Returns:
            horplane
        """

        # Get the points of data in a dataframe
        df = self.get_plane_of_points(
                                  x1_resolution=x_resolution, 
                                  x2_resolution=y_resolution, 
                                  normal_vector='x',
                                  x3_value=x_loc,
                                  x1_bounds=x_bounds,
                                  x2_bounds=y_bounds)

        # Compute and return the cutplane
        return CutPlane(df)

    def get_y_plane(self, y_loc,
            x_resolution=200, 
            y_resolution=200, 
            x_bounds=None,
            y_bounds=None):
        """
        Get a horizontal cut through plane at a specific height

        Args:
            height (float): height of cut plane, defaults to hub-height
                Defaults to Hub-height.
            x1_resolution (float, optional): output array resolution.
                Defaults to 200.
            x2_resolution (float, optional): output array resolution.
                Defaults to 200.
            x1_bounds (tuple, optional): limits of output array.
                Defaults to None.
            x2_bounds (tuple, optional): limits of output array.
                Defaults to None.

        Returns:
            horplane
        """

        # Get the points of data in a dataframe
        df = self.get_plane_of_points(
                                  x1_resolution=x_resolution, 
                                  x2_resolution=y_resolution, 
                                  normal_vector='y',
                                  x3_value=y_loc,
                                  x1_bounds=x_bounds,
                                  x2_bounds=y_bounds)

        # Compute and return the cutplane
        return CutPlane(df)


    def get_flow_data(self, resolution=None, grid_spacing=10, velocity_deficit = False):
        """
        Generate FlowData object corresponding to the floris instance.

        #TODO disambiguate between resolution and grid spacing.

        Args:
            resolution (float, optional): resolution of output data.
                Only used for wake models that require spatial
                resolution (e.g. curl). Defaults to None.
            grid_spacing (int, optional): resolution of output data.
                Defaults to 10.
            velocity_deficit (bool, optional): normalizes velocity with 
                respect to initial flow field velocity to show percent 
                velocity deficit (%).

        Returns:
            :py:class:`floris.tools.flow_data.FlowData`: FlowData object
        """

        if resolution is None:
            if not self.floris.farm.flow_field.wake.velocity_model.requires_resolution:
                print('Assuming grid with spacing %d' % grid_spacing)
                xmin, xmax, ymin, ymax, zmin, zmax = self.floris.farm.flow_field.domain_bounds
                resolution = Vec3(1 + (xmax - xmin) / grid_spacing,
                                  1 + (ymax - ymin) / grid_spacing,
                                  1 + (zmax - zmin) / grid_spacing)
            else:
                print('Assuming model resolution')
                resolution = self.floris.farm.flow_field.wake.velocity_model.model_grid_resolution

        # Get a copy for the flow field so don't change underlying grid points
        flow_field = copy.deepcopy(self.floris.farm.flow_field)

        if flow_field.wake.velocity_model.requires_resolution and \
            flow_field.wake.velocity_model.model_grid_resolution != resolution:
            print(
                "WARNING: The current wake velocity model contains a required grid resolution;"
            )
            print(
                "    The Resolution given to FlorisInterface.get_flow_field is ignored."
            )
            resolution = flow_field.wake.velocity_model.model_grid_resolution
        flow_field.reinitialize_flow_field(with_resolution=resolution)
        print(resolution)
        flow_field.calculate_wake()

        order = "f"
        x = flow_field.x.flatten(order=order)
        y = flow_field.y.flatten(order=order)
        z = flow_field.z.flatten(order=order)

        u = flow_field.u.flatten(order=order)
        v = flow_field.v.flatten(order=order)
        w = flow_field.w.flatten(order=order)

        # find percent velocity deficit
        if velocity_deficit == True:
            u = abs(u - flow_field.u_initial.flatten(order=order))/flow_field.u_initial.flatten(order=order) * 100
            v = abs(v - flow_field.v_initial.flatten(order=order))/flow_field.v_initial.flatten(order=order) * 100
            w = abs(w - flow_field.w_initial.flatten(order=order))/flow_field.w_initial.flatten(order=order) * 100

        # Determine spacing, dimensions and origin
        unique_x = np.sort(np.unique(x))
        unique_y = np.sort(np.unique(y))
        unique_z = np.sort(np.unique(z))
        spacing = Vec3(unique_x[1] - unique_x[0], unique_y[1] - unique_y[0],
                       unique_z[1] - unique_z[0])
        dimensions = Vec3(len(unique_x), len(unique_y), len(unique_z))
        origin = Vec3(0.0, 0.0, 0.0)
        return FlowData(x,
                        y,
                        z,
                        u,
                        v,
                        w,
                        spacing=spacing,
                        dimensions=dimensions,
                        origin=origin)

    def get_yaw_angles(self):
        """
        Report yaw angles of wind turbines from instance of floris.

        Returns:
            yaw_angles (np.array): wind turbine yaw angles.
        """
        yaw_angles = [
            turbine.yaw_angle
            for turbine in self.floris.farm.turbine_map.turbines
        ]
        return yaw_angles

    def get_farm_power(self, include_unc=False, unc_pmfs=None, unc_options=None, no_wake=False,  use_turbulence_correction = False):
        """
        Report wind plant power from instance of floris. Optionally includes uncertainty
        in wind direction and yaw position when determining power. Uncertainty is included
        by computing the mean wind farm power for a distribution of wind direction and yaw
        position deviations from the original wind direction and yaw angles.

        Args:
            include_unc (bool): If True, uncertainty in wind direction
                and/or yaw position is included when determining wind farm power.
                Defaults to False.
            unc_pmfs (dictionary, optional): A dictionary containing optional
                probability mass functions describing the distribution of wind
                direction and yaw position deviations when wind direction and/or
                yaw position uncertainty is included in the power calculations.
                Contains the following key-value pairs:

                -   **wd_unc**: A numpy array containing wind direction deviations
                    from the original wind direction.
                -   **wd_unc_pmf**: A numpy array containing the probability of
                    each wind direction deviation in **wd_unc** occuring.
                -   **yaw_unc**: A numpy array containing yaw angle deviations
                    from the original yaw angles.
                -   **yaw_unc_pmf**: A numpy array containing the probability of
                    each yaw angle deviation in **yaw_unc** occuring.

                Defaults to None, in which case default PMFs are calculated using
                values provided in **unc_options**.
            unc_options (disctionary, optional): A dictionary containing values used
                to create normally-distributed, zero-mean probability mass functions
                describing the distribution of wind direction and yaw position
                deviations when wind direction and/or yaw position uncertainty is
                included. This argument is only used when **unc_pmfs** is None and
                contains the following key-value pairs:

                -   **std_wd**: A float containing the standard deviation of the wind
                        direction deviations from the original wind direction.
                -   **std_yaw**: A float containing the standard deviation of the yaw
                        angle deviations from the original yaw angles.
                -   **pmf_res**: A float containing the resolution in degrees of the
                        wind direction and yaw angle PMFs.
                -   **pdf_cutoff**: A float containing the cumulative distribution
                    function value at which the tails of the PMFs are truncated.

                Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw': 1.75,
                'pmf_res': 1.0, 'pdf_cutoff': 0.995}.
            no_wake: (bool, optional): When *True* updates the turbine
                quantities without calculating the wake or adding the
                wake to the flow field. Defaults to False.
            use_turbulence_correction: (bool, optional): When *True* uses a
                turbulence parameter to adjust power output calculations. 
                Defaults to False.

        Returns:
            plant_power (float): sum of wind turbine powers.
        """
        for turbine in self.floris.farm.turbines: turbine.use_turbulence_correction = use_turbulence_correction
        if include_unc:
            if (unc_options is None) & (unc_pmfs is None):
                unc_options = {'std_wd': 4.95, 'std_yaw': 1.75, \
                            'pmf_res': 1.0, 'pdf_cutoff': 0.995}

            if unc_pmfs is None:
                # create normally distributed wd and yaw uncertaitny pmfs
                if unc_options['std_wd'] > 0:
                    wd_bnd = int(np.ceil(norm.ppf(unc_options['pdf_cutoff'], \
                                    scale=unc_options['std_wd'])/unc_options['pmf_res']))
                    wd_unc = np.linspace(-1*wd_bnd*unc_options['pmf_res'], \
                                    wd_bnd*unc_options['pmf_res'],2*wd_bnd+1)
                    wd_unc_pmf = norm.pdf(wd_unc,scale=unc_options['std_wd'])
                    wd_unc_pmf = wd_unc_pmf / np.sum(wd_unc_pmf) # normalize so sum = 1.0
                else:
                    wd_unc = np.zeros(1)
                    wd_unc_pmf = np.ones(1)

                if unc_options['std_yaw'] > 0:
                    yaw_bnd = int(np.ceil(norm.ppf(unc_options['pdf_cutoff'], \
                                    scale=unc_options['std_yaw'])/unc_options['pmf_res']))
                    yaw_unc = np.linspace(-1*yaw_bnd*unc_options['pmf_res'], \
                                    yaw_bnd*unc_options['pmf_res'],2*yaw_bnd+1)
                    yaw_unc_pmf = norm.pdf(yaw_unc,scale=unc_options['std_yaw'])
                    yaw_unc_pmf = yaw_unc_pmf / np.sum(yaw_unc_pmf) # normalize so sum = 1.0
                else:
                    yaw_unc = np.zeros(1)
                    yaw_unc_pmf = np.ones(1)

                unc_pmfs = {'wd_unc': wd_unc, 'wd_unc_pmf': wd_unc_pmf, \
                            'yaw_unc': yaw_unc, 'yaw_unc_pmf': yaw_unc_pmf}

            mean_farm_power = 0.
            wd_orig = np.array(self.floris.farm.wind_map.input_direction)

            yaw_angles = self.get_yaw_angles()

            for i_wd,delta_wd in enumerate(unc_pmfs['wd_unc']):
                self.reinitialize_flow_field(wind_direction=wd_orig+delta_wd)

                for i_yaw,delta_yaw in enumerate(unc_pmfs['yaw_unc']):
                    mean_farm_power = mean_farm_power + unc_pmfs['wd_unc_pmf'][i_wd] \
                        * unc_pmfs['yaw_unc_pmf'][i_yaw] \
                        * self.get_farm_power_for_yaw_angle(list(np.array(yaw_angles)+delta_yaw),no_wake=no_wake)

            # reinitialize with original values
            self.reinitialize_flow_field(wind_direction=wd_orig)
            self.calculate_wake(yaw_angles=yaw_angles,no_wake=no_wake)
            return mean_farm_power
        else:
            turb_powers = [turbine.power for turbine in self.floris.farm.turbines]
            return np.sum(turb_powers)

    def get_turbine_power(self, include_unc=False, unc_pmfs=None, unc_options=None, no_wake=False, use_turbulence_correction = False):
        """
        Report power from each wind turbine from instance of floris.

        Args:
            include_unc (bool): If True, uncertainty in wind direction
                and/or yaw position is included when determining wind farm power.
                Defaults to False.
            unc_pmfs (dictionary, optional): A dictionary containing optional
                probability mass functions describing the distribution of wind
                direction and yaw position deviations when wind direction and/or
                yaw position uncertainty is included in the power calculations.
                Contains the following key-value pairs:

                -   **wd_unc**: A numpy array containing wind direction deviations
                    from the original wind direction.
                -   **wd_unc_pmf**: A numpy array containing the probability of
                    each wind direction deviation in **wd_unc** occuring.
                -   **yaw_unc**: A numpy array containing yaw angle deviations
                    from the original yaw angles.
                -   **yaw_unc_pmf**: A numpy array containing the probability of
                    each yaw angle deviation in **yaw_unc** occuring.

                Defaults to None, in which case default PMFs are calculated using
                values provided in **unc_options**.
            unc_options (disctionary, optional): A dictionary containing values used
                to create normally-distributed, zero-mean probability mass functions
                describing the distribution of wind direction and yaw position
                deviations when wind direction and/or yaw position uncertainty is
                included. This argument is only used when **unc_pmfs** is None and
                contains the following key-value pairs:

                -   **std_wd**: A float containing the standard deviation of the wind
                        direction deviations from the original wind direction.
                -   **std_yaw**: A float containing the standard deviation of the yaw
                        angle deviations from the original yaw angles.
                -   **pmf_res**: A float containing the resolution in degrees of the
                        wind direction and yaw angle PMFs.
                -   **pdf_cutoff**: A float containing the cumulative distribution
                    function value at which the tails of the PMFs are truncated.

                Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw': 1.75,
                'pmf_res': 1.0, 'pdf_cutoff': 0.995}.
            no_wake: (bool, optional): When *True* updates the turbine
                quantities without calculating the wake or adding the
                wake to the flow field. Defaults to False.
            use_turbulence_correction: (bool, optional): When *True* uses a
                turbulence parameter to adjust power output calculations. 
                Defaults to False.

        Returns:
            turb_powers (np.array): power produced by each wind turbine.
        """
        for turbine in self.floris.farm.turbines: turbine.use_turbulence_correction = use_turbulence_correction
        if include_unc:
            if (unc_options is None) & (unc_pmfs is None):
                unc_options = {'std_wd': 4.95, 'std_yaw': 1.75, \
                            'pmf_res': 1.0, 'pdf_cutoff': 0.995}

            if unc_pmfs is None:
                # create normally distributed wd and yaw uncertaitny pmfs
                if unc_options['std_wd'] > 0:
                    wd_bnd = int(np.ceil(norm.ppf(unc_options['pdf_cutoff'], \
                                    scale=unc_options['std_wd'])/unc_options['pmf_res']))
                    wd_unc = np.linspace(-1*wd_bnd*unc_options['pmf_res'], \
                                    wd_bnd*unc_options['pmf_res'],2*wd_bnd+1)
                    wd_unc_pmf = norm.pdf(wd_unc,scale=unc_options['std_wd'])
                    wd_unc_pmf = wd_unc_pmf / np.sum(wd_unc_pmf) # normalize so sum = 1.0
                else:
                    wd_unc = np.zeros(1)
                    wd_unc_pmf = np.ones(1)

                if unc_options['std_yaw'] > 0:
                    yaw_bnd = int(np.ceil(norm.ppf(unc_options['pdf_cutoff'], \
                                    scale=unc_options['std_yaw'])/unc_options['pmf_res']))
                    yaw_unc = np.linspace(-1*yaw_bnd*unc_options['pmf_res'], \
                                    yaw_bnd*unc_options['pmf_res'],2*yaw_bnd+1)
                    yaw_unc_pmf = norm.pdf(yaw_unc,scale=unc_options['std_yaw'])
                    yaw_unc_pmf = yaw_unc_pmf / np.sum(yaw_unc_pmf) # normalize so sum = 1.0
                else:
                    yaw_unc = np.zeros(1)
                    yaw_unc_pmf = np.ones(1)

                unc_pmfs = {'wd_unc': wd_unc, 'wd_unc_pmf': wd_unc_pmf, \
                            'yaw_unc': yaw_unc, 'yaw_unc_pmf': yaw_unc_pmf}

            mean_farm_power = np.zeros(len(self.floris.farm.turbines))
            wd_orig = np.array(self.floris.farm.wind_map.input_direction)

            yaw_angles = self.get_yaw_angles()

            for i_wd,delta_wd in enumerate(unc_pmfs['wd_unc']):
                self.reinitialize_flow_field(wind_direction= wd_orig + delta_wd)

                for i_yaw,delta_yaw in enumerate(unc_pmfs['yaw_unc']):
                    self.calculate_wake(yaw_angles=list(np.array(yaw_angles)+delta_yaw),no_wake=no_wake)
                    mean_farm_power = mean_farm_power + unc_pmfs['wd_unc_pmf'][i_wd] \
                        * unc_pmfs['yaw_unc_pmf'][i_yaw] \
                        * np.array([turbine.power for turbine in self.floris.farm.turbines])

            # reinitialize with original values
            self.reinitialize_flow_field(wind_direction=wd_orig)
            self.calculate_wake(yaw_angles=yaw_angles,no_wake=no_wake)
            return list(mean_farm_power)
        else:
            turb_powers = [
                turbine.power
                for turbine in self.floris.farm.turbines
            ]
            return turb_powers


    def get_turbine_ct(self):
        """
        Report thrust coefficient from each wind turbine from instance of floris.

        Returns:
            turb_ct_array (np.array): thrust coefficient for each wind turbine.
        """
        turb_ct_array = [
            turbine.Ct
            for turbine in self.floris.farm.flow_field.turbine_map.turbines
        ]
        return turb_ct_array

        # calculate the power under different yaw angles
    def get_farm_power_for_yaw_angle(self, yaw_angles, include_unc=False, unc_pmfs=None, unc_options=None, no_wake=False):
        """
        Assign yaw angles to turbines, calculate wake, report power

        Args:
            yaw_angles (np.array): yaw to apply to each turbine
            no_wake: (bool, optional): When *True* updates the turbine
                quantities without calculating the wake or adding the
                wake to the flow field. Defaults to False.

        Returns:
            power (float): wind plant power. #TODO negative? in kW?
        """

        self.calculate_wake(yaw_angles=yaw_angles,no_wake=no_wake)

        return self.get_farm_power(include_unc=include_unc, unc_pmfs=unc_pmfs, unc_options=unc_options)

    def get_farm_AEP(self, wd, ws, freq, yaw=None):
        AEP_sum = 0

        for i in range(len(wd)):
            self.reinitialize_flow_field(
                wind_direction=[wd[i]], wind_speed=[ws[i]])
            if yaw is None:
                self.calculate_wake()
            else:
                self.calculate_wake(yaw[i])

            AEP_sum = AEP_sum + self.get_farm_power()*freq[i]*8760
        return AEP_sum

    @property
    def layout_x(self):
        """
        Wind turbine coordinate information.

        Returns:
            layout_x (np.array): Wind turbine x-coordinate (east-west).
        """
        coords = self.floris.farm.flow_field.turbine_map.coords
        layout_x = np.zeros(len(coords))
        for i, coord in enumerate(coords):
            layout_x[i] = coord.x1
        return layout_x

    @property
    def layout_y(self):
        """
        Wind turbine coordinate information.

        Returns:
            layout_y (np.array): Wind turbine y-coordinate (east-west).
        """
        coords = self.floris.farm.flow_field.turbine_map.coords
        layout_y = np.zeros(len(coords))
        for i, coord in enumerate(coords):
            layout_y[i] = coord.x2
        return layout_y 
            
    def TKE_to_TI(self, turbulence_kinetic_energy, wind_speed):
        """
        Converts a list of turbulence kinetic energy values to  
            turbulence intensity.
        
        Args:
            turbulence_kinetic_energy (list): values of turbulence kinetic
                energy in untis of meters squared per second squared.
            wind_speed (list): measurements of wind speed in meters per second. 
                
        Returns:
            turbulence_intensity (list): converted turbulence intensity 
                values expressed in decimal fractions.       
        """
        turbulence_intensity = [(np.sqrt((2/3) * turbulence_kinetic_energy[i]))/wind_speed[i] for i in range(len(turbulence_kinetic_energy))]
       
        return turbulence_intensity
    
    def set_rotor_diameter(self, rotor_diameter):
        """
        Assign rotor diameter to turbines.

        Args:
            rotor_diameter: the rotor diameter(s) to be 
                applied to the turbines in meters. 
        """
        if isinstance(rotor_diameter, float) or isinstance(rotor_diameter, int):
            rotor_diameter = [rotor_diameter] * len(self.floris.farm.turbines)
        else:
            rotor_diameter = rotor_diameter
        for i,turbine in enumerate(self.floris.farm.turbines):
            turbine.rotor_diameter = rotor_diameter[i]

    # TODO
    # Comment this out until sure we'll need it
    # def get_velocity_at_point(self, points, initial = False):
    #     """
    #     Get waked velocity at specified points in the flow field. 

    #     Args:
    #         points (np.array): x, y and z coordinates of specified point(s)
    #             where flow_field velocity should be reported.
    #         initial(bool, optional): if set to True, the initial velocity of 
    #             the flow field is returned instead of the waked velocity.
    #             Defaults to False.

    #     Returns:
    #         velocity (list): flow field velocity at specified grid point(s), in m/s.
    #     """
    #     xp, yp, zp = points[0], points[1], points[2]
    #     x, y, z = self.floris.farm.flow_field.x, self.floris.farm.flow_field.y, self.floris.farm.flow_field.z
    #     velocity = self.floris.farm.flow_field.u
    #     initial_velocity = self.floris.farm.wind_map.grid_wind_speed
    #     pVel = []
    #     for i in range(len(xp)):
    #         xloc, yloc, zloc =np.array(x == xp[i]),np.array(y == yp[i]),np.array(z == zp[i])
    #         loc = np.logical_and(np.logical_and(xloc, yloc) == True, zloc == True)
    #         if initial == True: pVel.append(np.mean(initial_velocity[loc]))
    #         else: pVel.append(np.mean(velocity[loc]))

    #     return pVel
