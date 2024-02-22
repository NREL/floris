from pathlib import Path

import numpy as np

from floris.tools import FlorisInterface
from floris.tools.wind_data import WindDataBase
from floris.type_dec import (
    floris_array_converter,
    NDArrayBool,
    NDArrayFloat,
)


class UncertaintyInterface(FlorisInterface):
    def __init__(
        self,
        configuration: dict | str | Path,
        wd_resolution=1.0,  # Degree
        ws_resolution=1.0,  # m/s
        ti_resolution=0.01,
        yaw_resolution=1.0,  # Degree
        power_setpoint_resolution=100,  # kW
        wd_std=3.0,
        wd_sample_points=None,
        verbose=False,
    ):

        # Save these inputs
        self.wd_resolution = wd_resolution
        self.ws_resolution = ws_resolution
        self.ti_resolution = ti_resolution
        self.yaw_resolution = yaw_resolution
        self.power_setpoint_resolution = power_setpoint_resolution
        self.wd_std = wd_std
        self.verbose= verbose

        # If wd_sample_points, default to 1 and 2 std
        if wd_sample_points is None:
            wd_sample_points = [-2 * wd_std, -1 * wd_std, 0, wd_std, 2 * wd_std]

        self.wd_sample_points = wd_sample_points
        self.n_sample_points = len(self.wd_sample_points)

        # Get the weights
        self.weights = self._get_weights(self.wd_std, self.wd_sample_points)


        # Call base init function
        super().__init__(configuration)  # Call the parent's __init__

    def set(self,
        wind_speeds: list[float] | NDArrayFloat | None = None,
        wind_directions: list[float] | NDArrayFloat | None = None,
        wind_shear: float | None = None,
        wind_veer: float | None = None,
        reference_wind_height: float | None = None,
        turbulence_intensities: list[float] | NDArrayFloat | None = None,
        air_density: float | None = None,
        layout_x: list[float] | NDArrayFloat | None = None,
        layout_y: list[float] | NDArrayFloat | None = None,
        turbine_type: list | None = None,
        turbine_library_path: str | Path | None = None,
        solver_settings: dict | None = None,
        heterogenous_inflow_config=None,
        wind_data: type[WindDataBase] | None = None,
        yaw_angles: NDArrayFloat | list[float] | None = None,
        power_setpoints: NDArrayFloat | list[float] | list[float, None] | None = None,
        disable_turbines: NDArrayBool | list[bool] | None = None,):

        # Call the base function
        super().set(wind_speeds=wind_speeds,
                    wind_directions=wind_directions,
                    wind_shear=wind_shear,
                    wind_veer=wind_veer,
                    reference_wind_height=reference_wind_height,
                    turbulence_intensities=turbulence_intensities,
                    air_density=air_density,
                    layout_x=layout_x,
                    layout_y=layout_y,
                    turbine_type=turbine_type,
                    turbine_library_path=turbine_library_path,
                    solver_settings=solver_settings,
                    heterogenous_inflow_config=heterogenous_inflow_config,
                    wind_data=wind_data,
                    yaw_angles=yaw_angles,
                    power_setpoints=power_setpoints,
                    disable_turbines=disable_turbines,
                    )

        self._set_uncertain()

    def _set_uncertain(
        self,
    ):
        """
        Sets the underlying wind direction (wd), wind speed (ws), turbulence intensity (ti),
          yaw angle, and power setpoint for unique conditions, accounting for uncertainties.

        Args:
            wd_resolution (float, optional): Resolution for wind direction in degrees.
                Defaults to 1.0.
            ws_resolution (float, optional): Resolution for wind speed in m/s. Defaults to 1.0.
            ti_resolution (float, optional): Resolution for turbulence intensity. Defaults to 0.025.
            yaw_resolution (float, optional): Resolution for yaw angle in degrees. Defaults to 1.0.
            power_setpoint_resolution (int, optional): Resolution for power setpoint in kW. Defaults
                to 100.
            wd_std (float, optional): Standard deviation for wind direction. Defaults to 3.0.
            wd_sample_points (list, optional): Sample points for wind direction,
                defaulting to [-2 * wd_std, -1 * wd_std, 0, wd_std, 2 * wd_std]. Defaults to None.
            verbose (bool, optional): Whether to display information about sizes. Defaults to False.
        """


        # Grab the unexpanded values of all arrays
        # These original dimensions are what is returned
        self.wind_directions_unexpanded = self.floris.flow_field.wind_directions
        self.wind_speeds_unexpanded = self.floris.flow_field.wind_speeds
        self.turbulence_intensities_unexpanded = self.floris.flow_field.turbulence_intensities
        self.yaw_angles_unexpanded = self.floris.farm.yaw_angles
        self.power_setpoints_unexpanded = self.floris.farm.power_setpoints
        self.n_unexpanded = len(self.wind_directions_unexpanded)

        # Combine into the complete unexpanded_inputs
        self.unexpanded_inputs = np.hstack(
            (
                self.wind_directions_unexpanded[:, np.newaxis],
                self.wind_speeds_unexpanded[:, np.newaxis],
                self.turbulence_intensities_unexpanded[:, np.newaxis],
                self.yaw_angles_unexpanded,
                self.power_setpoints_unexpanded,
            )
        )

        # Get the rounded inputs
        self.rounded_inputs = self._get_rounded_inputs(
            self.unexpanded_inputs,
            self.wd_resolution,
            self.ws_resolution,
            self.ti_resolution,
            self.yaw_resolution,
            self.power_setpoint_resolution,
        )

        # Get the expanded inputs
        self._expanded_wind_directions = self._expand_wind_directions(
            self.rounded_inputs, self.wd_sample_points
        )
        self.n_expanded = self._expanded_wind_directions.shape[0]

        # Get the unique inputs
        self.unique_inputs, self.map_to_expanded_inputs = self._get_unique_inputs(
            self._expanded_wind_directions
        )
        self.n_unique = self.unique_inputs.shape[0]

        # Display info on sizes
        if self.verbose:
            print(f"Original num rows: {self.n_unexpanded}")
            print(f"Expanded num rows: {self.n_expanded}")
            print(f"Unique num rows: {self.n_unique}")

        # Now set the underlying wd/ws/ti/yaw/setpoint to check only the unique conditions
        super().set(
            wind_directions=self.unique_inputs[:, 0],
            wind_speeds=self.unique_inputs[:, 1],
            turbulence_intensities=self.unique_inputs[:, 2],
            yaw_angles=self.unique_inputs[:, 3 : 3 + self.floris.farm.n_turbines],
            power_setpoints=self.unique_inputs[:, 3 + self.floris.farm.n_turbines :],
        )

    def get_turbine_powers(self):
        """Calculates the power at each turbine in the wind farm.

        This method calculates the power at each turbine in the wind farm, considering
        the underlying turbine powers and applying a weighted sum to handle uncertainty.

        Returns:
            NDArrayFloat: An array containing the powers at each turbine for each finde.

        """

        # First call the underlying function
        unique_turbine_powers = super().get_turbine_powers()

        # Expand back to the expanded value
        expanded_turbine_powers = unique_turbine_powers[self.map_to_expanded_inputs]

        # Reshape the weights array to make it compatible with broadcasting
        weights_reshaped = self.weights[:, np.newaxis]

        # Reshape expanded_turbine_powers into blocks
        blocks = np.reshape(
            expanded_turbine_powers,
            (self.n_unexpanded, self.n_sample_points, self.floris.farm.n_turbines),
            order="F",
        )

        # Multiply each block by the corresponding weight
        weighted_blocks = blocks * weights_reshaped

        # Sum the blocks along the second axis
        result = np.sum(weighted_blocks, axis=1)

        return result

    def get_farm_power(
        self,
        turbine_weights=None,
    ):
        """
        Report wind plant power from instance of floris with uncertainty.

        Args:
            turbine_weights (NDArrayFloat | list[float] | None, optional):
                weighing terms that allow the user to emphasize power at
                particular turbines and/or completely ignore the power
                from other turbines. This is useful when, for example, you are
                modeling multiple wind farms in a single floris object. If you
                only want to calculate the power production for one of those
                farms and include the wake effects of the neighboring farms,
                you can set the turbine_weights for the neighboring farms'
                turbines to 0.0. The array of turbine powers from floris
                is multiplied with this array in the calculation of the
                objective function. If None, this  is an array with all values
                1.0 and with shape equal to (n_findex, n_turbines).
                Defaults to None.

        Returns:
            float: Sum of wind turbine powers in W.
        """

        if turbine_weights is None:
            # Default to equal weighing of all turbines when turbine_weights is None
            turbine_weights = np.ones(
                (
                    self.n_unexpanded,
                    self.floris.farm.n_turbines,
                )
            )
        elif len(np.shape(turbine_weights)) == 1:
            # Deal with situation when 1D array is provided
            turbine_weights = np.tile(
                turbine_weights,
                (self.n_unexpanded, 1),
            )

        # Calculate all turbine powers and apply weights
        turbine_powers = self.get_turbine_powers()
        turbine_powers = np.multiply(turbine_weights, turbine_powers)

        return np.sum(turbine_powers, axis=1)

    def get_farm_AEP(
        self,
        freq,
        cut_in_wind_speed=0.001,
        cut_out_wind_speed=None,
        turbine_weights=None,
        no_wake=False,
    ) -> float:
        """
        Estimate annual energy production (AEP) for distributions of wind speed, wind
        direction, frequency of occurrence, and yaw offset.

        Args:
            freq (NDArrayFloat): NumPy array with shape (n_findex)
                with the frequencies of each wind direction and
                wind speed combination. These frequencies should typically sum
                up to 1.0 and are used to weigh the wind farm power for every
                condition in calculating the wind farm's AEP.
            cut_in_wind_speed (float, optional): Wind speed in m/s below which
                any calculations are ignored and the wind farm is known to
                produce 0.0 W of power. Note that to prevent problems with the
                wake models at negative / zero wind speeds, this variable must
                always have a positive value. Defaults to 0.001 [m/s].
            cut_out_wind_speed (float, optional): Wind speed above which the
                wind farm is known to produce 0.0 W of power. If None is
                specified, will assume that the wind farm does not cut out
                at high wind speeds. Defaults to None.
            turbine_weights (NDArrayFloat | list[float] | None, optional):
                weighing terms that allow the user to emphasize power at
                particular turbines and/or completely ignore the power
                from other turbines. This is useful when, for example, you are
                modeling multiple wind farms in a single floris object. If you
                only want to calculate the power production for one of those
                farms and include the wake effects of the neighboring farms,
                you can set the turbine_weights for the neighboring farms'
                turbines to 0.0. The array of turbine powers from floris
                is multiplied with this array in the calculation of the
                objective function. If None, this  is an array with all values
                1.0 and with shape equal to (n_findex,
                n_turbines). Defaults to None.
            no_wake: (bool, optional): When *True* updates the turbine
                quantities without calculating the wake or adding the wake to
                the flow field. This can be useful when quantifying the loss
                in AEP due to wakes. Defaults to *False*.


        Returns:
            float:
                The Annual Energy Production (AEP) for the wind farm in
                watt-hours.
        """

        # Verify dimensions of the variable "freq"
        if np.shape(freq)[0] != self.n_unexpanded:
            raise UserWarning(
                "'freq' should be a one-dimensional array with dimensions (self.n_unexpanded). "
                f"Given shape is {np.shape(freq)}"
            )

        # Check if frequency vector sums to 1.0. If not, raise a warning
        if np.abs(np.sum(freq) - 1.0) > 0.001:
            self.logger.warning(
                "WARNING: The frequency array provided to get_farm_AEP() does not sum to 1.0."
            )

        # Copy the full wind speed array from the floris object and initialize
        # the the farm_power variable as an empty array.
        wind_directions = np.array(self.wind_directions_unexpanded, copy=True)
        wind_speeds = np.array(self.wind_speeds_unexpanded, copy=True)
        farm_power = np.zeros_like(wind_directions)

        # Determine which wind speeds we must evaluate
        conditions_to_evaluate = wind_speeds >= cut_in_wind_speed
        if cut_out_wind_speed is not None:
            conditions_to_evaluate = conditions_to_evaluate & (wind_speeds < cut_out_wind_speed)

        # Evaluate the conditions in floris
        if np.any(conditions_to_evaluate):
            wind_speeds_subset = wind_speeds[conditions_to_evaluate]
            wind_directions_subset = wind_directions[conditions_to_evaluate]
            self.set(
                wind_speeds=wind_speeds_subset,
                wind_directions=wind_directions_subset,
            )

            if no_wake:
                self.run_no_wake()
            else:
                self.run()
            farm_power[conditions_to_evaluate] = self.get_farm_power(
                turbine_weights=turbine_weights
            )

        # Finally, calculate AEP in GWh
        aep = np.sum(np.multiply(freq, farm_power) * 365 * 24)

        # Reset the FLORIS object to the full wind speed array
        self.set(wind_speeds=wind_speeds, wind_directions=wind_directions)

        return aep

    def get_farm_AEP_with_wind_data(
        self,
        wind_data,
        cut_in_wind_speed=0.001,
        cut_out_wind_speed=None,
        turbine_weights=None,
        no_wake=False,
    ) -> float:
        """
        Estimate annual energy production (AEP) for distributions of wind speed, wind
        direction, frequency of occurrence, and yaw offset.

        Args:
            wind_data: (type(WindDataBase)): TimeSeries or WindRose object containing
                the wind conditions over which to calculate the AEP. Should match the wind_data
                object passed to reinitialize().
            cut_in_wind_speed (float, optional): Wind speed in m/s below which
                any calculations are ignored and the wind farm is known to
                produce 0.0 W of power. Note that to prevent problems with the
                wake models at negative / zero wind speeds, this variable must
                always have a positive value. Defaults to 0.001 [m/s].
            cut_out_wind_speed (float, optional): Wind speed above which the
                wind farm is known to produce 0.0 W of power. If None is
                specified, will assume that the wind farm does not cut out
                at high wind speeds. Defaults to None.
            turbine_weights (NDArrayFloat | list[float] | None, optional):
                weighing terms that allow the user to emphasize power at
                particular turbines and/or completely ignore the power
                from other turbines. This is useful when, for example, you are
                modeling multiple wind farms in a single floris object. If you
                only want to calculate the power production for one of those
                farms and include the wake effects of the neighboring farms,
                you can set the turbine_weights for the neighboring farms'
                turbines to 0.0. The array of turbine powers from floris
                is multiplied with this array in the calculation of the
                objective function. If None, this  is an array with all values
                1.0 and with shape equal to (n_findex,
                n_turbines). Defaults to None.
            no_wake: (bool, optional): When *True* updates the turbine
                quantities without calculating the wake or adding the wake to
                the flow field. This can be useful when quantifying the loss
                in AEP due to wakes. Defaults to *False*.

        Returns:
            float:
                The Annual Energy Production (AEP) for the wind farm in
                watt-hours.
        """

        # Verify the wind_data object matches FLORIS' initialization
        if wind_data.n_findex != self.n_unexpanded:
            raise ValueError("WindData object findex not length n_unexpanded")

        # Get freq directly from wind_data
        freq = wind_data.unpack_freq()

        return self.get_farm_AEP(
            freq,
            cut_in_wind_speed=cut_in_wind_speed,
            cut_out_wind_speed=cut_out_wind_speed,
            turbine_weights=turbine_weights,
            no_wake=no_wake,
        )

    def _get_rounded_inputs(
        self,
        input_array,
        wd_resolution=1.0,  # Degree
        ws_resolution=1.0,  # m/s
        ti_resolution=0.025,
        yaw_resolution=1.0,  # Degree
        power_setpoint_resolution=100,  # kW
    ):
        """
        Round the input array  specified resolutions.

        Parameters:
            input_array (numpy.ndarray): An array of shape (n, 5)  with columns
                                        for wind direction (wd), wind speed (ws),
                                        turbulence intensity (tu),
                                        yaw angle (yaw), and power setpoint.
            wd_resolution (float): Resolution for rounding wind direction in degrees.
                Default is 1.0 degree.
            ws_resolution (float): Resolution for rounding wind speed in m/s. Default is 1.0 m/s.
            ti_resolution (float): Resolution for rounding turbulence intensity. Default is 0.1.
            yaw_resolution (float): Resolution for rounding yaw angle in degrees.
                Default is 1.0 degree.
            power_setpoint_resolution (int): Resolution for rounding power setpoint in kW.
                Default is 100 kW.

        Returns:
            numpy.ndarray: A rounded array of wind turbine parameters with
                    the same shape as input_array,
                    where each parameter is rounded to the specified resolution.
        """

        # input_array is a nx5 numpy array whose columns are wd, ws, tu, yaw, power_setpoint
        # round each column by the respective resolution
        rounded_input_array = np.copy(input_array)
        rounded_input_array[:, 0] = (
            np.round(rounded_input_array[:, 0] / wd_resolution) * wd_resolution
        )
        rounded_input_array[:, 1] = (
            np.round(rounded_input_array[:, 1] / ws_resolution) * ws_resolution
        )
        rounded_input_array[:, 2] = (
            np.round(rounded_input_array[:, 2] / ti_resolution) * ti_resolution
        )
        rounded_input_array[:, 3] = (
            np.round(rounded_input_array[:, 3] / yaw_resolution) * yaw_resolution
        )
        rounded_input_array[:, 4] = (
            np.round(rounded_input_array[:, 4] / power_setpoint_resolution)
            * power_setpoint_resolution
        )

        return rounded_input_array

    def _expand_wind_directions(self, input_array, wd_sample_points):
        """
        Expand wind direction data.

        Args:
            input_array (numpy.ndarray): 2D numpy array of shape (m, n)
            representing wind direction data,
                where m is the number of data points and n is the number of features.
                The first column
                represents wind direction.
            wd_sample_points (list): List of integers representing
            wind direction sample points.

        Returns:
            numpy.ndarray: Expanded wind direction data as a 2D numpy array
                of shape (m * p, n), where
                p is the number of sample points.

        Raises:
            ValueError: If wd_sample_points does not have an odd length or
                if the middle element is not 0.

        This function takes wind direction data and expands it
        by perturbing the wind direction column
        based on a list of sample points. It vertically stacks
        copies of the input array with the wind
        direction column perturbed by each sample point, ensuring
        the resultant values are within the range
        of 0 to 360.
        """

        # Check if wd_sample_points is odd-length and the middle element is 0
        if len(wd_sample_points) % 2 != 1:
            raise ValueError("wd_sample_points must have an odd length.")
        if wd_sample_points[len(wd_sample_points) // 2] != 0:
            raise ValueError("The middle element of wd_sample_points must be 0.")

        num_samples = len(wd_sample_points)
        num_rows = input_array.shape[0]

        # Create an array to hold the expanded data
        output_array = np.zeros((num_rows * num_samples, input_array.shape[1]))

        # Repeat each row of input_array for each sample point
        for i in range(num_samples):
            start_idx = i * num_rows
            end_idx = start_idx + num_rows
            output_array[start_idx:end_idx, :] = input_array.copy()

            # Perturb the wd column by the current sample point
            output_array[start_idx:end_idx, 0] = (
                output_array[start_idx:end_idx, 0] + wd_sample_points[i]
            ) % 360

        return output_array

    def _get_unique_inputs(self, input_array):
        """
        Finds unique rows in the input numpy array and constructs a mapping array
        to reconstruct the input array from the unique rows.

        Args:
            input_array (numpy.ndarray): Input array of shape (m, n).

        Returns:
            tuple: A tuple containing:
                numpy.ndarray: An array of unique rows found in the input_array, of shape (r, n),
                            where r <= m.
                numpy.ndarray: A 1D array of indices mapping each row of the input_array
                            to the corresponding row in the unique_inputs array.
                            It represents how to reconstruct the input_array from the unique rows.
        """

        unique_inputs, indices, map_to_expanded_inputs = np.unique(
            input_array, axis=0, return_index=True, return_inverse=True
        )


        return unique_inputs, map_to_expanded_inputs

    def _get_weights(self, wd_std, wd_sample_points):
        """Generates weights based on a Gaussian distribution sampled at specific x-locations.

        Args:
            wd_std (float): The standard deviation of the Gaussian distribution.
            wd_sample_points (array-like): The x-locations where the Gaussian function is sampled.

        Returns:
            numpy.ndarray: An array of weights, generated using a Gaussian distribution with mean 0
                and standard deviation wd_std, sampled at the specified x-locations.
                The weights are normalized so that they sum to 1.

        """

        # Calculate the Gaussian function values at sample points
        gaussian_values = np.exp(-(np.array(wd_sample_points) ** 2) / (2 * wd_std**2))

        # Normalize the Gaussian values to get the weights
        weights = gaussian_values / np.sum(gaussian_values)

        return weights

    def copy(self):
        raise NotImplementedError("Copy not implemented for UncertaintyInterface")

    def get_plane_of_points(
        self,
        **_,
    ):
        raise NotImplementedError("get_plane_of_points not implemented for UncertaintyInterface")

    def calculate_horizontal_plane(
        self,
        **_,
    ):
        raise NotImplementedError(
            "calculate_horizontal_plane not implemented for UncertaintyInterface"
        )

    def calculate_cross_plane(
        self,
        **_,
    ):
        raise NotImplementedError("calculate_cross_plane not implemented for UncertaintyInterface")

    def calculate_y_plane(
        self,
        **_,
    ):
        raise NotImplementedError("calculate_y_plane not implemented for UncertaintyInterface")

    def check_wind_condition_for_viz(self, **_,):
        raise NotImplementedError(
            "check_wind_condition_for_viz not implemented for UncertaintyInterface"
        )

    def get_turbine_thrust_coefficients(self):
        raise NotImplementedError(
            "get_turbine_thrust_coefficients not implemented for UncertaintyInterface"
        )

    def get_turbine_ais(self):
        raise NotImplementedError("get_turbine_ais not implemented for UncertaintyInterface")

    def sample_flow_at_points(self, **_,):
        raise NotImplementedError("sample_flow_at_points not implemented for UncertaintyInterface")

    def sample_velocity_deficit_profiles(
        self,
        **_,
    ):
        raise NotImplementedError(
            "sample_velocity_deficit_profiles not implemented for UncertaintyInterface"
        )
