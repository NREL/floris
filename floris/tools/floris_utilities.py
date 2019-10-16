# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
from floris.simulation import Floris
from floris.simulation import TurbineMap
from .flow_data import FlowData
from ..utilities import Vec3
import copy
from scipy.stats import norm


class FlorisInterface():
    """
    The interface between a FLORIS instance and the wfc tools
    """

    def __init__(self, input_file=None, input_dict=None):
        if input_file is None and input_dict is None:
            raise ValueError('Input file or dictionary must be supplied')
        self.input_file = input_file
        self.floris = Floris(input_file=input_file, input_dict=input_dict)

    def calculate_wake(self, yaw_angles=None, no_wake=False):
        """
        Wrapper to the floris flow field calculate_wake method

        Args:
            yaw_angles (np.array, optional): Turbine yaw angles.
                Defaults to None.
            no_wake: (bool, optional): When *True* updates the turbine
                quantities without calculating the wake or adding the
                wake to the flow field. Defaults to False.
        """

        if yaw_angles is not None:
            self.floris.farm.set_yaw_angles(yaw_angles)

        self.floris.farm.flow_field.calculate_wake(no_wake=no_wake)

    def reinitialize_flow_field(self,
                                wind_speed=None,
                                wind_direction=None,
                                wind_shear=None,
                                wind_veer=None,
                                turbulence_intensity=None,
                                air_density=None,
                                wake=None,
                                layout_array=None,
                                with_resolution=None):
        """
        Wrapper to
        :py:meth:`floris.simlulation.flow_field.reinitialize_flow_field`.
        All input values are used to update the flow_field instance.

        Args:
            wind_speed (float, optional): background wind speed.
                Defaults to None.
            wind_direction (float, optional): background wind direction.
                Defaults to None.
            wind_shear (float, optional): shear exponent.
                Defaults to None.
            wind_veer (float, optional): direction change over rotor.
                Defaults to None.
            turbulence_intensity (float, optional): background turbulence
                intensity. Defaults to None.
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

        # Build turbine map (convenience layer for user)
        if layout_array is not None:
            turbine_map = TurbineMap(
                layout_array[0], layout_array[1], \
                [copy.deepcopy(self.floris.farm.turbines[0]) \
                for ii in range(len(layout_array[0]))])
        else:
            turbine_map = None

        self.floris.farm.flow_field.reinitialize_flow_field(
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            wind_shear=wind_shear,
            wind_veer=wind_veer,
            turbulence_intensity=turbulence_intensity,
            air_density=air_density,
            wake=wake,
            turbine_map=turbine_map,
            with_resolution=with_resolution)

    # Special case function for quick visualization of hub height
    def get_hub_height_flow_data(self,
                                 x_resolution=100,
                                 y_resolution=100,
                                 x_bounds=None,
                                 y_bounds=None):
        """
        Shortcut method to visualize flow field at hub height.

        Args:
            x_resolution (float, optional): output array resolution.
                Defaults to 100.
            y_resolution (float, optional): output array resolution.
                Defaults to 100.
            x_bounds (tuple, optional): limits of output array.
                Defaults to None.
            y_bounds (tuple, optional): limits of output array.
                Defaults to None.

        Returns:
            :py:class:`floris.tools.flow_data.FlowData`: FlowData object at hub
                height.
        """
        if self.floris.farm.flow_field.wake.velocity_model.requires_resolution:
            raise Exception(
                'Not allowed for wake model %s ' %
                self.floris.farm.flow_field.wake.velocity_model.model_string)

        # Get a copy for the flow field so don't change underlying grid points
        flow_field = copy.deepcopy(self.floris.farm.flow_field)

        # If x and y bounds are not provided, use rules of thumb
        if x_bounds is None:
            coords = self.floris.farm.flow_field.turbine_map.coords
            max_diameter = self.floris.farm.flow_field.max_diameter
            x = [coord.x1 for coord in coords]
            x_bounds = (min(x) - 2 * max_diameter, max(x) + 10 * max_diameter)
        if y_bounds is None:
            coords = self.floris.farm.flow_field.turbine_map.coords
            max_diameter = self.floris.farm.flow_field.max_diameter
            y = [coord.x2 for coord in coords]
            y_bounds = (min(y) - 2 * max_diameter, max(y) + 2 * max_diameter)

        # Z_bounds is always hub-height
        hub_height = self.floris.farm.flow_field.turbine_map.turbines[
            0].hub_height
        bounds_to_set = (x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1],
                         hub_height - 5., hub_height + 5.)

        # Set new bounds
        flow_field.set_bounds(bounds_to_set=bounds_to_set)

        # Change the resolution
        flow_field.reinitialize_flow_field(
            with_resolution=Vec3(x_resolution, y_resolution, 3))

        # Calculate the wakes
        flow_field.calculate_wake()

        order = "f"
        x = flow_field.x.flatten(order=order)
        y = flow_field.y.flatten(order=order)
        z = flow_field.z.flatten(order=order)

        u = flow_field.u.flatten(order=order)
        v = flow_field.v.flatten(order=order)
        w = flow_field.w.flatten(order=order)

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

    def get_flow_data(self, resolution=None, grid_spacing=10):
        """
        Generate FlowData object corresponding to the floris instance.

        #TODO disambiguate between resolution and grid spacing.

        Args:
            resolution (float, optional): resolution of output data.
                Only used for wake models that require spatial
                resolution (e.g. curl). Defaults to None.
            grid_spacing (int, optional): resolution of output data.
                Defaults to 10.

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

    def get_farm_power(self, include_unc=False, unc_pmfs=None, unc_options=None, no_wake=False):
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

        Returns:
            plant_power (float): sum of wind turbine powers.
        """
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
            wd_orig = self.floris.farm.wind_direction

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

    def get_turbine_power(self, include_unc=False, unc_pmfs=None, unc_options=None, no_wake=False):
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

        Returns:
            turb_powers (np.array): power produced by each wind turbine.
        """
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
            wd_orig = self.floris.farm.wind_direction

            yaw_angles = self.get_yaw_angles()

            for i_wd,delta_wd in enumerate(unc_pmfs['wd_unc']):
                self.reinitialize_flow_field(wind_direction=wd_orig+delta_wd)

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

    def get_farm_AEP(self, wd, ws, freq):
        AEP_sum = 0

        for i in range(len(wd)):
            self.reinitialize_flow_field(
                wind_direction=wd[i], wind_speed=ws[i])
            self.calculate_wake()

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
