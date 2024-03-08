from __future__ import annotations

from abc import abstractmethod

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from floris.type_dec import NDArrayFloat


class WindDataBase:
    """
    Super class that WindRose and TimeSeries inherit from, enforcing the implementation of
    unpack() on the child classes and providing the general functions unpack_for_reinitialize() and
    unpack_freq().
    """

    @abstractmethod
    def unpack(self):
        """
        Placeholder for child classes of WindDataBase, which each need to implement the unpack()
        method.
        """
        raise NotImplementedError("unpack() not implemented on {0}".format(self.__class__.__name__))

    def unpack_for_reinitialize(self):
        """
        Return only the variables need for FlorisInterface.reinitialize
        """
        (
            wind_directions_unpack,
            wind_speeds_unpack,
            ti_table_unpack,
            _,
            _,
            heterogenous_inflow_config,
        ) = self.unpack()

        return (
            wind_directions_unpack,
            wind_speeds_unpack,
            ti_table_unpack,
            heterogenous_inflow_config,
        )

    def unpack_freq(self):
        """Unpack frequency weighting"""

        (
            _,
            _,
            _,
            freq_table_unpack,
            _,
            _,
        ) = self.unpack()

        return freq_table_unpack

    def check_heterogenous_inflow_config_by_wd(self, heterogenous_inflow_config_by_wd):
        """
        Check that the heterogenous_inflow_config_by_wd dictionary is properly formatted

        Args:
            heterogenous_inflow_config_by_wd (dict): A dictionary containing the following keys:
                * 'speed_multipliers': A 2D NumPy array (size num_wd x num_points)
                     of speed multipliers.
                * 'wind_directions': A 1D NumPy array (size num_wd) of wind directions (degrees).
                * 'x': A 1D NumPy array (size num_points) of x-coordinates (meters).
                * 'y': A 1D NumPy array (size num_points) of y-coordinates (meters).
        """
        if heterogenous_inflow_config_by_wd is not None:
            if not isinstance(heterogenous_inflow_config_by_wd, dict):
                raise TypeError("heterogenous_inflow_config_by_wd must be a dictionary")
            if "speed_multipliers" not in heterogenous_inflow_config_by_wd:
                raise ValueError(
                    "heterogenous_inflow_config_by_wd must contain a key 'speed_multipliers'"
                )
            if "wind_directions" not in heterogenous_inflow_config_by_wd:
                raise ValueError(
                    "heterogenous_inflow_config_by_wd must contain a key 'wind_directions'"
                )
            if "x" not in heterogenous_inflow_config_by_wd:
                raise ValueError("heterogenous_inflow_config_by_wd must contain a key 'x'")
            if "y" not in heterogenous_inflow_config_by_wd:
                raise ValueError("heterogenous_inflow_config_by_wd must contain a key 'y'")

    def check_heterogenous_inflow_config(self, heterogenous_inflow_config):
        """
        Check that the heterogenous_inflow_config dictionary is properly formatted

        Args:
            heterogenous_inflow_config (dict): A dictionary containing the following keys:
                * 'speed_multipliers': A 2D NumPy array (size n_findex x num_points)
                      of speed multipliers.
                * 'x': A 1D NumPy array (size num_points) of x-coordinates (meters).
                * 'y': A 1D NumPy array (size num_points) of y-coordinates (meters).
        """
        if heterogenous_inflow_config is not None:
            if not isinstance(heterogenous_inflow_config, dict):
                raise TypeError("heterogenous_inflow_config_by_wd must be a dictionary")
            if "speed_multipliers" not in heterogenous_inflow_config:
                raise ValueError(
                    "heterogenous_inflow_config must contain a key 'speed_multipliers'"
                )
            if "x" not in heterogenous_inflow_config:
                raise ValueError("heterogenous_inflow_config must contain a key 'x'")
            if "y" not in heterogenous_inflow_config:
                raise ValueError("heterogenous_inflow_config must contain a key 'y'")

    def get_speed_multipliers_by_wd(self, heterogenous_inflow_config_by_wd, wind_directions):
        """
        Processes heterogenous inflow configuration data to generate a speed multiplier array
        aligned with the wind directions. Accounts for the cyclical nature of wind directions.
        Args:
            heterogenous_inflow_config_by_wd (dict): A dictionary containing the following keys:
                * 'speed_multipliers': A 2D NumPy array (size num_wd x num_points)
                     of speed multipliers.
                * 'wind_directions': A 1D NumPy array (size num_wd) of wind directions (degrees).
                * 'x': A 1D NumPy array (size num_points) of x-coordinates (meters).
                * 'y': A 1D NumPy array (size num_points) of y-coordinates (meters).

            wind_directions (np.array): Wind directions to map onto
        Returns:
                numpy.ndarray: A 2D NumPy array (size n_findex x n) of speed multipliers
                            Each row corresponds to a wind direction,
                            with speed multipliers selected
                            based on the closest matching wind direction in 'het_wd'.
        """

        # Extract data from the configuration dictionary
        speed_multipliers = np.array(heterogenous_inflow_config_by_wd["speed_multipliers"])
        het_wd = np.array(heterogenous_inflow_config_by_wd["wind_directions"])

        # Confirm 0th dimension of speed_multipliers == len(het_wd)
        if len(het_wd) != speed_multipliers.shape[0]:
            raise ValueError(
                "The legnth of het_wd must equal the number of rows speed_multipliers"
                "Within the heterogenous_inflow_config_by_wd dictionary"
            )

        # Calculate closest wind direction indices (accounting for angles)
        angle_diffs = np.abs(wind_directions[:, None] - het_wd)
        min_angle_diffs = np.minimum(angle_diffs, 360 - angle_diffs)
        closest_wd_indices = np.argmin(min_angle_diffs, axis=1)

        # Construct the output array using the calculated indices
        return speed_multipliers[closest_wd_indices]

    def get_heterogenous_inflow_config(self, heterogenous_inflow_config_by_wd, wind_directions):
        # If heterogenous_inflow_config_by_wd is None, return None
        if heterogenous_inflow_config_by_wd is None:
            return None

        # If heterogenous_inflow_config_by_wd is not None, then process it
        # Build the n-findex version of the het map
        speed_multipliers = self.get_speed_multipliers_by_wd(
            heterogenous_inflow_config_by_wd, wind_directions
        )
        # Return heterogenous_inflow_config
        return {
            "speed_multipliers": speed_multipliers,
            "x": heterogenous_inflow_config_by_wd["x"],
            "y": heterogenous_inflow_config_by_wd["y"],
        }


class WindRose(WindDataBase):
    """
    The WindRose class is used to drive FLORIS and optimization operations in
    which the inflow is characterized by the frequency of binned wind speed and
    wind direction values.  Turbulence intensities are defined as a function of
    wind direction and wind speed.

    Args:
        wind_directions: NumPy array of wind directions (NDArrayFloat).
        wind_speeds: NumPy array of wind speeds (NDArrayFloat).
        ti_table: Turbulence intensity table for binned wind direction, wind
            speed values (float, NDArrayFloat).  Can be an array with dimensions
            (n_wind_directions, n_wind_speeds) or a single float value.  If a
            single float value is provided, the turbulence intensity is assumed
            to be constant across all wind directions and wind speeds.
        freq_table: Frequency table for binned wind direction, wind speed
            values (NDArrayFloat, optional).   Must have dimension
            (n_wind_directions, n_wind_speeds).  Defaults to None in which case
            uniform frequency of all bins is assumed.
        value_table: Value table for binned wind direction, wind
            speed values (NDArrayFloat, optional).  Must have dimension
            (n_wind_directions, n_wind_speeds).  Defaults to None in which case
            uniform values are assumed.  Value can be used to weight power in
            each bin to compute the total value of the energy produced
        compute_zero_freq_occurrence: Flag indicating whether to compute zero
            frequency occurrences (bool, optional).  Defaults to False.
        heterogenous_inflow_config_by_wd (dict, optional): A dictionary containing the following
            keys. Defaults to None.
            * 'speed_multipliers': A 2D NumPy array (size num_wd x num_points)
                    of speed multipliers.
            * 'wind_directions': A 1D NumPy array (size num_wd) of wind directions (degrees).
            * 'x': A 1D NumPy array (size num_points) of x-coordinates (meters).
            * 'y': A 1D NumPy array (size num_points) of y-coordinates (meters).

    """

    def __init__(
        self,
        wind_directions: NDArrayFloat,
        wind_speeds: NDArrayFloat,
        ti_table: float | NDArrayFloat,
        freq_table: NDArrayFloat | None = None,
        value_table: NDArrayFloat | None = None,
        compute_zero_freq_occurrence: bool = False,
        heterogenous_inflow_config_by_wd: dict | None = None,
    ):
        if not isinstance(wind_directions, np.ndarray):
            raise TypeError("wind_directions must be a NumPy array")

        if not isinstance(wind_speeds, np.ndarray):
            raise TypeError("wind_speeds must be a NumPy array")

        # Save the wind speeds and directions
        self.wind_directions = wind_directions
        self.wind_speeds = wind_speeds

        # Check if ti_table is a single float value
        if isinstance(ti_table, float):
            self.ti_table = np.full((len(wind_directions), len(wind_speeds)), ti_table)

        # Otherwise confirm the dimensions and then save it
        else:
            if not ti_table.shape[0] == len(wind_directions):
                raise ValueError("ti_table first dimension must equal len(wind_directions)")
            if not ti_table.shape[1] == len(wind_speeds):
                raise ValueError("ti_table second dimension must equal len(wind_speeds)")
            self.ti_table = ti_table

        # If freq_table is not None, confirm it has correct dimension,
        # otherwise initialize to uniform probability
        if freq_table is not None:
            if not freq_table.shape[0] == len(wind_directions):
                raise ValueError("freq_table first dimension must equal len(wind_directions)")
            if not freq_table.shape[1] == len(wind_speeds):
                raise ValueError("freq_table second dimension must equal len(wind_speeds)")
            self.freq_table = freq_table
        else:
            self.freq_table = np.ones((len(wind_directions), len(wind_speeds)))

        # Normalize freq table
        self.freq_table = self.freq_table / np.sum(self.freq_table)

        # If value_table is not None, confirm it has correct dimension,
        # otherwise initialize to all ones
        if value_table is not None:
            if not value_table.shape[0] == len(wind_directions):
                raise ValueError("value_table first dimension must equal len(wind_directions)")
            if not value_table.shape[1] == len(wind_speeds):
                raise ValueError("value_table second dimension must equal len(wind_speeds)")
        self.value_table = value_table

        # Save whether zero occurrence cases should be computed
        # First check if the ti_table contains any nan values (which would occur for example
        # if generated by the TimeSeries to WindRose conversion for wind speeds and directions
        # that were not present in the original time series)  In this case, raise an error
        if compute_zero_freq_occurrence:
            if np.isnan(self.ti_table).any():
                raise ValueError(
                    "ti_table contains nan values.  (This is likely the result of "
                    " unsed wind speeds and directions in the original time series.)"
                    "  Cannot compute zero frequency occurrences."
                )
        self.compute_zero_freq_occurrence = compute_zero_freq_occurrence

        # Check that heterogenous_inflow_config_by_wd is a dictionary with keys:
        # speed_multipliers, wind_directions, x and y
        self.check_heterogenous_inflow_config_by_wd(heterogenous_inflow_config_by_wd)

        # Then save
        self.heterogenous_inflow_config_by_wd = heterogenous_inflow_config_by_wd

        # Build the gridded and flatten versions
        self._build_gridded_and_flattened_version()

    def _build_gridded_and_flattened_version(self):
        """
        Given the wind direction and speed array, build the gridded versions
        covering all combinations, and then flatten versions which put all
        combinations into 1D array
        """
        # Gridded wind speed and direction
        self.wd_grid, self.ws_grid = np.meshgrid(
            self.wind_directions, self.wind_speeds, indexing="ij"
        )

        # Flat wind speed and direction
        self.wd_flat = self.wd_grid.flatten()
        self.ws_flat = self.ws_grid.flatten()

        # Flat frequency table
        self.freq_table_flat = self.freq_table.flatten()

        # Flat TI table
        self.ti_table_flat = self.ti_table.flatten()

        # value table
        if self.value_table is not None:
            self.value_table_flat = self.value_table.flatten()
        else:
            self.value_table_flat = None

        # Set mask to non-zero frequency cases depending on compute_zero_freq_occurrence
        if self.compute_zero_freq_occurrence:
            # If computing zero freq occurrences, then this is all True
            self.non_zero_freq_mask = [True for i in range(len(self.freq_table_flat))]
        else:
            self.non_zero_freq_mask = self.freq_table_flat > 0.0

        # N_findex should only be the calculated cases
        self.n_findex = np.sum(self.non_zero_freq_mask)

    def unpack(self):
        """
        Unpack the flattened versions of the matrices and return the values
        accounting for the non_zero_freq_mask
        """

        # The unpacked versions start as the flat version of each
        wind_directions_unpack = self.wd_flat.copy()
        wind_speeds_unpack = self.ws_flat.copy()
        freq_table_unpack = self.freq_table_flat.copy()
        ti_table_unpack = self.ti_table_flat.copy()

        # Now mask thes values according to self.non_zero_freq_mask
        wind_directions_unpack = wind_directions_unpack[self.non_zero_freq_mask]
        wind_speeds_unpack = wind_speeds_unpack[self.non_zero_freq_mask]
        freq_table_unpack = freq_table_unpack[self.non_zero_freq_mask]
        ti_table_unpack = ti_table_unpack[self.non_zero_freq_mask]

        # Now get unpacked value table
        if self.value_table_flat is not None:
            value_table_unpack = self.value_table_flat[self.non_zero_freq_mask].copy()
        else:
            value_table_unpack = None

        # If heterogenous_inflow_config_by_wd is not None, then update
        # heterogenous_inflow_config to match wind_directions_unpack
        if self.heterogenous_inflow_config_by_wd is not None:
            heterogenous_inflow_config = self.get_heterogenous_inflow_config(
                self.heterogenous_inflow_config_by_wd, wind_directions_unpack
            )
        else:
            heterogenous_inflow_config = None

        return (
            wind_directions_unpack,
            wind_speeds_unpack,
            ti_table_unpack,
            freq_table_unpack,
            value_table_unpack,
            heterogenous_inflow_config,
        )

    def resample_wind_rose(self, wd_step=None, ws_step=None):
        """
        Resamples the wind rose by by wd_step and/or ws_step

        Args:
            wd_step: Step size for wind direction resampling (float, optional).
            ws_step: Step size for wind speed resampling (float, optional).

        Returns:
            WindRose: Resampled wind rose based on the provided or default step sizes.

        Notes:
            - Returns a resampled version of the wind rose using new `ws_step` and `wd_step`.
            - Uses the bin weights feature in TimeSeries to resample the wind rose.
            - If `ws_step` or `wd_step` is not specified, it uses the current values.
        """
        if ws_step is None:
            if len(self.wind_speeds) >= 2:
                ws_step = self.wind_speeds[1] - self.wind_speeds[0]
            else:  # wind rose will have only a single wind speed, and we assume a ws_step of 1
                ws_step = 1.0
        if wd_step is None:
            if len(self.wind_directions) >= 2:
                wd_step = self.wind_directions[1] - self.wind_directions[0]
            else:  # wind rose will have only a single wind direction, and we assume a wd_step of 1
                wd_step = 1.0

        # Pass the flat versions of each quantity to build a TimeSeries model
        time_series = TimeSeries(
            self.wd_flat,
            self.ws_flat,
            self.ti_table_flat,
            self.value_table_flat,
            self.heterogenous_inflow_config_by_wd,
        )

        # Now build a new wind rose using the new steps
        return time_series.to_wind_rose(
            wd_step=wd_step, ws_step=ws_step, bin_weights=self.freq_table_flat
        )

    def plot_wind_rose(
        self,
        ax=None,
        color_map="viridis_r",
        wd_step=15.0,
        ws_step=5.0,
        legend_kwargs={},
    ):
        """
        This method creates a wind rose plot showing the frequency of occurrence
        of the specified wind direction and wind speed bins. If no axis is
        provided, a new one is created.

        **Note**: Based on code provided by Patrick Murphy from the University
        of Colorado Boulder.

        Args:
            ax (:py:class:`matplotlib.pyplot.axes`, optional): The figure axes
                on which the wind rose is plotted. Defaults to None.
            color_map (str, optional): Colormap to use. Defaults to 'viridis_r'.
            wd_step: Step size for wind direction  (float, optional).
            ws_step: Step size for wind speed  (float, optional).
            legend_kwargs (dict, optional): Keyword arguments to be passed to
                ax.legend().

        Returns:
            :py:class:`matplotlib.pyplot.axes`: A figure axes object containing
            the plotted wind rose.
        """

        # Get a resampled wind_rose
        wind_rose_resample = self.resample_wind_rose(wd_step, ws_step)
        wd_bins = wind_rose_resample.wind_directions
        ws_bins = wind_rose_resample.wind_speeds
        freq_table = wind_rose_resample.freq_table

        # Set up figure
        if ax is None:
            _, ax = plt.subplots(subplot_kw={"polar": True})

        # Get a color array
        color_array = cm.get_cmap(color_map, len(ws_bins))

        for wd_idx, wd in enumerate(wd_bins):
            rects = []
            freq_table_sub = freq_table[wd_idx, :].flatten()
            for ws_idx, ws in reversed(list(enumerate(ws_bins))):
                plot_val = freq_table_sub[:ws_idx].sum()
                rects.append(
                    ax.bar(
                        np.radians(wd),
                        plot_val,
                        width=0.9 * np.radians(wd_step),
                        color=color_array(ws_idx),
                        edgecolor="k",
                    )
                )

        # Configure the plot
        ax.legend(reversed(rects), ws_bins, **legend_kwargs)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_theta_zero_location("N")
        ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 4))
        ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

        return ax

    def assign_ti_using_wd_ws_function(self, func):
        """
        Use the passed in function to assign new values to turbulence_intensities

        Args:
            func (function): Function which accepts wind_directions as its
                first argument and wind_speeds as second argument and returns
                turbulence_intensities
        """
        self.ti_table = func(self.wd_grid, self.ws_grid)
        self._build_gridded_and_flattened_version()

    def assign_ti_using_IEC_method(self, Iref=0.07, offset=3.8):
        """
        Define TI as a function of wind speed by specifying an Iref and offset
        value as in the normal turbulence model in the IEC 61400-1 standard

        Args:
            Iref (float): Reference turbulence level, defined as the expected
                value of TI at 15 m/s. Default = 0.07. Note this value is
                lower than the values of Iref for turbulence classes A, B, and
                C in the IEC standard (0.16, 0.14, and 0.12, respectively), but
                produces TI values more in line with those typically used in
                FLORIS. When the default Iref and offset are used, the TI at
                8 m/s is 8.6%.
            offset (float): Offset value to equation. Default = 3.8, as defined
                in the IEC standard to give the expected value of TI for
                each wind speed.
        """
        if (Iref < 0) or (Iref > 1):
            raise ValueError("Iref must be >= 0 and <=1")

        def iref_func(wind_directions, wind_speeds):
            sigma_1 = Iref * (0.75 * wind_speeds + offset)
            return sigma_1 / wind_speeds

        self.assign_ti_using_wd_ws_function(iref_func)

    def plot_ti_over_ws(
        self,
        ax=None,
        marker=".",
        ls="None",
        color="k",
    ):
        """
        Scatter plot the turbulence_intensities against wind_speeds

        Args:
            ax (:py:class:`matplotlib.pyplot.axes`, optional): The figure axes
                on which the wind rose is plotted. Defaults to None.
            plot_kwargs (dict, optional): Keyword arguments to be passed to
                ax.plot().

        Returns:
            :py:class:`matplotlib.pyplot.axes`: A figure axes object containing
            the plotted wind rose.
        """

        # Set up figure
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self.ws_flat, self.ti_table_flat * 100, marker=marker, ls=ls, color=color)
        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Turbulence Intensity (%)")
        ax.grid(True)


class WindTIRose(WindDataBase):
    """
    WindTIRose is similar to the WindRose class, but contains turbulence
    intensity as an additional wind rose dimension instead of being defined
    as a function of wind direction and wind speed. The class is used to drive
    FLORIS and optimization operations in which the inflow is characterized by
    the frequency of binned wind speed, wind direction, and turbulence intensity
    values.

    Args:
        wind_directions: NumPy array of wind directions (NDArrayFloat).
        wind_speeds: NumPy array of wind speeds (NDArrayFloat).
        turbulence_intensities: NumPy array of turbulence intensities (NDArrayFloat).
        freq_table: Frequency table for binned wind direction, wind speed, and
            turbulence intensity values (NDArrayFloat, optional). Must have
            dimension (n_wind_directions, n_wind_speeds, n_turbulence_intensities).
            Defaults to None in which case uniform frequency of all bins is
            assumed.
        value_table: Value table for binned wind direction, wind
            speed, and turbulence intensity values (NDArrayFloat, optional).
            Must have dimension (n_wind_directions, n_wind_speeds,
            n_turbulence_intensities). Defaults to None in which case uniform
            values are assumed. Value can be used to weight power in each bin
            to compute the total value of the energy produced.
        compute_zero_freq_occurrence: Flag indicating whether to compute zero
            frequency occurrences (bool, optional).  Defaults to False.
        heterogenous_inflow_config_by_wd (dict, optional): A dictionary containing the following
            keys. Defaults to None.
            * 'speed_multipliers': A 2D NumPy array (size num_wd x num_points)
                    of speed multipliers.
            * 'wind_directions': A 1D NumPy array (size num_wd) of wind directions (degrees).
            * 'x': A 1D NumPy array (size num_points) of x-coordinates (meters).
            * 'y': A 1D NumPy array (size num_points) of y-coordinates (meters).

    """

    def __init__(
        self,
        wind_directions: NDArrayFloat,
        wind_speeds: NDArrayFloat,
        turbulence_intensities: NDArrayFloat,
        freq_table: NDArrayFloat | None = None,
        value_table: NDArrayFloat | None = None,
        compute_zero_freq_occurrence: bool = False,
        heterogenous_inflow_config_by_wd: dict | None = None,
    ):
        if not isinstance(wind_directions, np.ndarray):
            raise TypeError("wind_directions must be a NumPy array")

        if not isinstance(wind_speeds, np.ndarray):
            raise TypeError("wind_speeds must be a NumPy array")

        if not isinstance(turbulence_intensities, np.ndarray):
            raise TypeError("turbulence_intensities must be a NumPy array")

        # Save the wind speeds and directions
        self.wind_directions = wind_directions
        self.wind_speeds = wind_speeds
        self.turbulence_intensities = turbulence_intensities

        # If freq_table is not None, confirm it has correct dimension,
        # otherwise initialize to uniform probability
        if freq_table is not None:
            if not freq_table.shape[0] == len(wind_directions):
                raise ValueError("freq_table first dimension must equal len(wind_directions)")
            if not freq_table.shape[1] == len(wind_speeds):
                raise ValueError("freq_table second dimension must equal len(wind_speeds)")
            if not freq_table.shape[2] == len(turbulence_intensities):
                raise ValueError(
                    "freq_table third dimension must equal len(turbulence_intensities)"
                )
            self.freq_table = freq_table
        else:
            self.freq_table = np.ones(
                (len(wind_directions), len(wind_speeds), len(turbulence_intensities))
            )

        # Normalize freq table
        self.freq_table = self.freq_table / np.sum(self.freq_table)

        # If value_table is not None, confirm it has correct dimension,
        # otherwise initialize to all ones
        if value_table is not None:
            if not value_table.shape[0] == len(wind_directions):
                raise ValueError("value_table first dimension must equal len(wind_directions)")
            if not value_table.shape[1] == len(wind_speeds):
                raise ValueError("value_table second dimension must equal len(wind_speeds)")
            if not value_table.shape[2] == len(turbulence_intensities):
                raise ValueError(
                    "value_table third dimension must equal len(turbulence_intensities)"
                )
        self.value_table = value_table

        # Check that heterogenous_inflow_config_by_wd is a dictionary with keys:
        # speed_multipliers, wind_directions, x and y
        self.check_heterogenous_inflow_config_by_wd(heterogenous_inflow_config_by_wd)

        # Then save
        self.heterogenous_inflow_config_by_wd = heterogenous_inflow_config_by_wd

        # Save whether zero occurrence cases should be computed
        self.compute_zero_freq_occurrence = compute_zero_freq_occurrence

        # Build the gridded and flatten versions
        self._build_gridded_and_flattened_version()

    def _build_gridded_and_flattened_version(self):
        """
        Given the wind direction, wind speed, and turbulence intensity array,
        build the gridded versions covering all combinations, and then flatten
        versions which put all combinations into 1D array
        """
        # Gridded wind speed and direction
        self.wd_grid, self.ws_grid, self.ti_grid = np.meshgrid(
            self.wind_directions, self.wind_speeds, self.turbulence_intensities, indexing="ij"
        )

        # Flat wind direction, wind speed, and turbulence intensity
        self.wd_flat = self.wd_grid.flatten()
        self.ws_flat = self.ws_grid.flatten()
        self.ti_flat = self.ti_grid.flatten()

        # Flat frequency table
        self.freq_table_flat = self.freq_table.flatten()

        # value table
        if self.value_table is not None:
            self.value_table_flat = self.value_table.flatten()
        else:
            self.value_table_flat = None

        # Set mask to non-zero frequency cases depending on compute_zero_freq_occurrence
        if self.compute_zero_freq_occurrence:
            # If computing zero freq occurrences, then this is all True
            self.non_zero_freq_mask = [True for i in range(len(self.freq_table_flat))]
        else:
            self.non_zero_freq_mask = self.freq_table_flat > 0.0

        # N_findex should only be the calculated cases
        self.n_findex = np.sum(self.non_zero_freq_mask)

    def unpack(self):
        """
        Unpack the flattened versions of the matrices and return the values
        accounting for the non_zero_freq_mask
        """

        # The unpacked versions start as the flat version of each
        wind_directions_unpack = self.wd_flat.copy()
        wind_speeds_unpack = self.ws_flat.copy()
        turbulence_intensities_unpack = self.ti_flat.copy()
        freq_table_unpack = self.freq_table_flat.copy()

        # Now mask thes values according to self.non_zero_freq_mask
        wind_directions_unpack = wind_directions_unpack[self.non_zero_freq_mask]
        wind_speeds_unpack = wind_speeds_unpack[self.non_zero_freq_mask]
        turbulence_intensities_unpack = turbulence_intensities_unpack[self.non_zero_freq_mask]
        freq_table_unpack = freq_table_unpack[self.non_zero_freq_mask]

        # Now get unpacked value table
        if self.value_table_flat is not None:
            value_table_unpack = self.value_table_flat[self.non_zero_freq_mask].copy()
        else:
            value_table_unpack = None

        # If heterogenous_inflow_config_by_wd is not None, then update
        # heterogenous_inflow_config to match wind_directions_unpack
        if self.heterogenous_inflow_config_by_wd is not None:
            heterogenous_inflow_config = self.get_heterogenous_inflow_config(
                self.heterogenous_inflow_config_by_wd, wind_directions_unpack
            )
        else:
            heterogenous_inflow_config = None

        return (
            wind_directions_unpack,
            wind_speeds_unpack,
            turbulence_intensities_unpack,
            freq_table_unpack,
            value_table_unpack,
            heterogenous_inflow_config,
        )

    def resample_wind_rose(self, wd_step=None, ws_step=None, ti_step=None):
        """
        Resamples the wind rose by by wd_step, ws_step, and/or ti_step

        Args:
            wd_step: Step size for wind direction resampling (float, optional).
            ws_step: Step size for wind speed resampling (float, optional).
            ti_step: Step size for turbulence intensity resampling (float, optional).

        Returns:
            WindRose: Resampled wind rose based on the provided or default step sizes.

        Notes:
            - Returns a resampled version of the wind rose using new `ws_step`,
                `wd_step`, and `ti_step`.
            - Uses the bin weights feature in TimeSeries to resample the wind rose.
            - If `ws_step`, `wd_step`, or `ti_step` are not specified, it uses
                the current values.
        """
        if ws_step is None:
            if len(self.wind_speeds) >= 2:
                ws_step = self.wind_speeds[1] - self.wind_speeds[0]
            else:  # wind rose will have only a single wind speed, and we assume a ws_step of 1
                ws_step = 1.0
        if wd_step is None:
            if len(self.wind_directions) >= 2:
                wd_step = self.wind_directions[1] - self.wind_directions[0]
            else:  # wind rose will have only a single wind direction, and we assume a wd_step of 1
                wd_step = 1.0
        if ti_step is None:
            if len(self.turbulence_intensities) >= 2:
                ti_step = self.turbulence_intensities[1] - self.turbulence_intensities[0]
            else:  # wind rose will have only a single TI, and we assume a ti_step of 1
                ti_step = 1.0

        # Pass the flat versions of each quantity to build a TimeSeries model
        time_series = TimeSeries(
            self.wd_flat,
            self.ws_flat,
            self.ti_flat,
            self.value_table_flat,
            self.heterogenous_inflow_config_by_wd,
        )

        # Now build a new wind rose using the new steps
        return time_series.to_wind_ti_rose(
            wd_step=wd_step, ws_step=ws_step, ti_step=ti_step, bin_weights=self.freq_table_flat
        )

    def plot_wind_rose(
        self,
        ax=None,
        wind_rose_var="ws",
        color_map="viridis_r",
        wd_step=15.0,
        wind_rose_var_step=None,
        legend_kwargs={},
    ):
        """
        This method creates a wind rose plot showing the frequency of occurrence
        of either the specified wind direction and wind speed bins or wind
        direction and turbulence intensity bins. If no axis is provided, a new
        one is created.

        **Note**: Based on code provided by Patrick Murphy from the University
        of Colorado Boulder.

        Args:
            ax (:py:class:`matplotlib.pyplot.axes`, optional): The figure axes
                on which the wind rose is plotted. Defaults to None.
            wind_rose_var (str, optional): The variable to display in the wind
                rose plot in addition to wind direction. If
                wind_rose_var = "ws", wind speed frequencies will be plotted.
                If wind_rose_var = "ti", turbulence intensity frequencies will
                be plotted. Defaults to "ws".
            color_map (str, optional): Colormap to use. Defaults to 'viridis_r'.
            wd_step (float, optional): Step size for wind direction. Defaults
                to 15 degrees.
            wind_rose_var_step (float, optional): Step size for other wind rose
                variable. Defaults to None. If unspecified, a value of 5 m/s
                will beused if wind_rose_var = "ws", and a value of 4% will be
                used if wind_rose_var = "ti".
            legend_kwargs (dict, optional): Keyword arguments to be passed to
                ax.legend().

        Returns:
            :py:class:`matplotlib.pyplot.axes`: A figure axes object containing
            the plotted wind rose.
        """

        if wind_rose_var not in {"ws", "ti"}:
            raise ValueError(
                'wind_rose_var must be either "ws" or "ti" for wind speed or turbulence intensity.'
            )

        # Get a resampled wind_rose
        if wind_rose_var == "ws":
            if wind_rose_var_step is None:
                wind_rose_var_step = 5.0
            wind_rose_resample = self.resample_wind_rose(wd_step, ws_step=wind_rose_var_step)
            var_bins = wind_rose_resample.wind_speeds
            freq_table = wind_rose_resample.freq_table.sum(2)  # sum along TI dimension
        else:  # wind_rose_var == "ti"
            if wind_rose_var_step is None:
                wind_rose_var_step = 0.04
            wind_rose_resample = self.resample_wind_rose(wd_step, ti_step=wind_rose_var_step)
            var_bins = wind_rose_resample.turbulence_intensities
            freq_table = wind_rose_resample.freq_table.sum(1)  # sum along wind speed dimension

        wd_bins = wind_rose_resample.wind_directions

        # Set up figure
        if ax is None:
            _, ax = plt.subplots(subplot_kw={"polar": True})

        # Get a color array
        color_array = cm.get_cmap(color_map, len(var_bins))

        for wd_idx, wd in enumerate(wd_bins):
            rects = []
            freq_table_sub = freq_table[wd_idx, :].flatten()
            for var_idx, ws in reversed(list(enumerate(var_bins))):
                plot_val = freq_table_sub[:var_idx].sum()
                rects.append(
                    ax.bar(
                        np.radians(wd),
                        plot_val,
                        width=0.9 * np.radians(wd_step),
                        color=color_array(var_idx),
                        edgecolor="k",
                    )
                )

        # Configure the plot
        ax.legend(reversed(rects), var_bins, **legend_kwargs)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_theta_zero_location("N")
        ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 4))
        ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

        return ax

    def plot_ti_over_ws(
        self,
        ax=None,
        marker=".",
        ls="-",
        color="k",
    ):
        """
        Plot the mean turbulence intensity against wind speed.

        Args:
            ax (:py:class:`matplotlib.pyplot.axes`, optional): The figure axes
                on which the wind rose is plotted. Defaults to None.
            plot_kwargs (dict, optional): Keyword arguments to be passed to
                ax.plot().

        Returns:
            :py:class:`matplotlib.pyplot.axes`: A figure axes object containing
            the plotted wind rose.
        """

        # TODO: Plot std. devs. of TI in addition to mean values

        # Set up figure
        if ax is None:
            _, ax = plt.subplots()

        # get mean TI for each wind speed by averaging along wind direction and
        # TI dimensions
        mean_ti_values = (self.ti_grid * self.freq_table).sum((0, 2)) / self.freq_table.sum((0, 2))

        ax.plot(self.wind_speeds, mean_ti_values * 100, marker=marker, ls=ls, color=color)
        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Mean Turbulence Intensity (%)")
        ax.grid(True)


class TimeSeries(WindDataBase):
    """
    The TimeSeries class is used to drive FLORIS and optimization operations in
    which the inflow is by a sequence of wind direction, wind speed and
    turbulence intensity values.  Each input of wind direction, wind speed, and
    turbulence intensity can be assigned as an array of values or a single value.
    At least one of wind_directions, wind_speeds, or turbulence_intensities must
    be an array.  If arrays are provided, they must be the same length as the
    other arrays or the single values.  If single values are provided, then an
    array of the same length as the other arrays will be created with the single
    value.

    Args:
        wind_directions (float, NDArrayFloat): Wind direction. Can be a single
            value or an array of values.
        wind_speeds (float, NDArrayFloat): Wind speed. Can be a single value or
            an array of values.
        turbulence_intensities (float, NDArrayFloat): Turbulence intensity. Can be
            a single value or an array of values.
        values (NDArrayFloat, optional): Values associated with each wind
            direction, wind speed, and turbulence intensity. Defaults to None.
        heterogenous_inflow_config_by_wd (dict, optional): A dictionary containing the following
            keys. Defaults to None.
            * 'speed_multipliers': A 2D NumPy array (size num_wd x num_points)
                    of speed multipliers.
            * 'wind_directions': A 1D NumPy array (size num_wd) of wind directions (degrees).
            * 'x': A 1D NumPy array (size num_points) of x-coordinates (meters).
            * 'y': A 1D NumPy array (size num_points) of y-coordinates (meters).
        heterogenous_inflow_config (dict, optional): A dictionary containing the following keys.
            Defaults to None.
            * 'speed_multipliers': A 2D NumPy array (size n_findex x num_points)
                    of speed multipliers.
            * 'x': A 1D NumPy array (size num_points) of x-coordinates (meters).
            * 'y': A 1D NumPy array (size num_points) of y-coordinates (meters).
    """

    def __init__(
        self,
        wind_directions: float | NDArrayFloat,
        wind_speeds: float | NDArrayFloat,
        turbulence_intensities: float | NDArrayFloat,
        values: NDArrayFloat | None = None,
        heterogenous_inflow_config_by_wd: dict | None = None,
        heterogenous_inflow_config: dict | None = None,
    ):
        # At least one of wind_directions, wind_speeds, or turbulence_intensities must be an array
        if (
            not isinstance(wind_directions, np.ndarray)
            and not isinstance(wind_speeds, np.ndarray)
            and not isinstance(turbulence_intensities, np.ndarray)
        ):
            raise TypeError(
                "At least one of wind_directions, wind_speeds, or "
                " turbulence_intensities must be a NumPy array"
            )

        # For each of wind_directions, wind_speeds, and turbulence_intensities provided as
        # an array, confirm they are the same length
        if isinstance(wind_directions, np.ndarray) and isinstance(wind_speeds, np.ndarray):
            if len(wind_directions) != len(wind_speeds):
                raise ValueError(
                    "wind_directions and wind_speeds must be the same length if provided as arrays"
                )

        if (
            isinstance(wind_directions, np.ndarray)
            and isinstance(turbulence_intensities, np.ndarray)
        ):
            if len(wind_directions) != len(turbulence_intensities):
                raise ValueError(
                    "wind_directions and turbulence_intensities must be "
                    "the same length if provided as arrays"
                )

        if isinstance(wind_speeds, np.ndarray) and isinstance(turbulence_intensities, np.ndarray):
            if len(wind_speeds) != len(turbulence_intensities):
                raise ValueError(
                    "wind_speeds and turbulence_intensities must be the "
                    "same length if provided as arrays"
                )

        # For each of wind_directions, wind_speeds, and turbulence_intensities
        # provided as a single value, set them
        # to be the same length as those passed in as arrays
        if isinstance(wind_directions, float):
            if isinstance(wind_speeds, np.ndarray):
                wind_directions = np.full(len(wind_speeds), wind_directions)
            elif isinstance(turbulence_intensities, np.ndarray):
                wind_directions = np.full(len(turbulence_intensities), wind_directions)

        if isinstance(wind_speeds, float):
            if isinstance(wind_directions, np.ndarray):
                wind_speeds = np.full(len(wind_directions), wind_speeds)
            elif isinstance(turbulence_intensities, np.ndarray):
                wind_speeds = np.full(len(turbulence_intensities), wind_speeds)

        if isinstance(turbulence_intensities, float):
            if isinstance(wind_directions, np.ndarray):
                turbulence_intensities = np.full(len(wind_directions), turbulence_intensities)
            elif isinstance(wind_speeds, np.ndarray):
                turbulence_intensities = np.full(len(wind_speeds), turbulence_intensities)

        # If values is not None, must be same length as wind_directions/wind_speeds/
        if values is not None:
            if len(wind_directions) != len(values):
                raise ValueError("wind_directions and values must be the same length")

        self.wind_directions = wind_directions
        self.wind_speeds = wind_speeds
        self.turbulence_intensities = turbulence_intensities
        self.values = values

        # Only one of heterogenous_inflow_config_by_wd and
        # heterogenous_inflow_config can be not None
        if heterogenous_inflow_config_by_wd is not None and heterogenous_inflow_config is not None:
            raise ValueError(
                "Only one of heterogenous_inflow_config_by_wd and heterogenous_inflow_config "
                "can be not None"
            )

        # if heterogenous_inflow_config is not None, then the speed_multipliers
        # must be the same length as wind_directions
        # in the 0th dimension
        if heterogenous_inflow_config is not None:
            if len(heterogenous_inflow_config["speed_multipliers"]) != len(wind_directions):
                raise ValueError("speed_multipliers must be the same length as wind_directions")

        # Check that heterogenous_inflow_config_by_wd is a dictionary with keys:
        # speed_multipliers, wind_directions, x and y
        self.check_heterogenous_inflow_config_by_wd(heterogenous_inflow_config_by_wd)
        self.check_heterogenous_inflow_config(heterogenous_inflow_config)

        # Then save
        self.heterogenous_inflow_config_by_wd = heterogenous_inflow_config_by_wd
        self.heterogenous_inflow_config = heterogenous_inflow_config

        # Record findex
        self.n_findex = len(self.wind_directions)

    def unpack(self):
        """
        Unpack the time series data in a manner consistent with wind rose unpack
        """

        # to match wind_rose, make a uniform frequency
        uniform_frequency = np.ones_like(self.wind_directions)
        uniform_frequency = uniform_frequency / uniform_frequency.sum()

        # If heterogenous_inflow_config_by_wd is not None, then update
        # heterogenous_inflow_config to match wind_directions_unpack
        if self.heterogenous_inflow_config_by_wd is not None:
            heterogenous_inflow_config = self.get_heterogenous_inflow_config(
                self.heterogenous_inflow_config_by_wd, self.wind_directions
            )
        else:
            heterogenous_inflow_config = self.heterogenous_inflow_config

        return (
            self.wind_directions,
            self.wind_speeds,
            self.turbulence_intensities,
            uniform_frequency,
            self.values,
            heterogenous_inflow_config,
        )

    def _wrap_wind_directions_near_360(self, wind_directions, wd_step):
        """
        Wraps the wind directions using `wd_step` to produce a wrapped version
        where values between [360 - wd_step/2.0, 360] get mapped to negative numbers
        for binning.

        Args:
            wind_directions (NDArrayFloat): NumPy array of wind directions.
            wd_step (float): Step size for wind direction.

        Returns:
            NDArrayFloat: Wrapped version of wind directions.

        """
        wind_directions_wrapped = wind_directions.copy()
        mask = wind_directions_wrapped >= 360 - wd_step / 2.0
        wind_directions_wrapped[mask] = wind_directions_wrapped[mask] - 360.0
        return wind_directions_wrapped

    def assign_ti_using_wd_ws_function(self, func):
        """
        Use the passed in function to new assign values to turbulence_intensities

        Args:
            func (function): Function which accepts wind_directions as its
                first argument and wind_speeds as second argument and returns
                turbulence_intensities
        """
        self.turbulence_intensities = func(self.wind_directions, self.wind_speeds)

    def assign_ti_using_IEC_method(self, Iref=0.07, offset=3.8):
        """
        Define TI as a function of wind speed by specifying an Iref and offset
        value as in the normal turbulence model in the IEC 61400-1 standard

        Args:
            Iref (float): Reference turbulence level, defined as the expected
                value of TI at 15 m/s. Default = 0.07. Note this value is
                lower than the values of Iref for turbulence classes A, B, and
                C in the IEC standard (0.16, 0.14, and 0.12, respectively), but
                produces TI values more in line with those typically used in
                FLORIS. When the default Iref and offset are used, the TI at
                8 m/s is 8.6%.
            offset (float): Offset value to equation. Default = 3.8, as defined
                in the IEC standard to give the expected value of TI for
                each wind speed.
        """
        if (Iref < 0) or (Iref > 1):
            raise ValueError("Iref must be >= 0 and <=1")

        def iref_func(wind_directions, wind_speeds):
            sigma_1 = Iref * (0.75 * wind_speeds + offset)
            return sigma_1 / wind_speeds

        self.assign_ti_using_wd_ws_function(iref_func)

    def to_wind_rose(
        self, wd_step=2.0, ws_step=1.0, wd_edges=None, ws_edges=None, bin_weights=None
    ):
        """
        Converts the TimeSeries data to a WindRose.

        Args:
            wd_step (float, optional): Step size for wind direction (default is 2.0).
            ws_step (float, optional): Step size for wind speed (default is 1.0).
            wd_edges (NDArrayFloat, optional): Custom wind direction edges. Defaults to None.
            ws_edges (NDArrayFloat, optional): Custom wind speed edges. Defaults to None.
            bin_weights (NDArrayFloat, optional): Bin weights for resampling.  Note these
                are primarily used by the resample resample_wind_rose function.
                Defaults to None.

        Returns:
            WindRose: A WindRose object based on the TimeSeries data.

        Notes:
            - If `wd_edges` is defined, it uses it to produce the bin centers.
            - If `wd_edges` is not defined, it determines `wd_edges` from the step and data.
            - If `ws_edges` is defined, it uses it for wind speed edges.
            - If `ws_edges` is not defined, it determines `ws_edges` from the step and data.
        """

        # If wd_edges is defined, then use it to produce the bin centers
        if wd_edges is not None:
            wd_step = wd_edges[1] - wd_edges[0]

            # use wd_step to produce a wrapped version of wind_directions
            wind_directions_wrapped = self._wrap_wind_directions_near_360(
                self.wind_directions, wd_step
            )

        # Else, determine wd_edges from the step and data
        else:
            wd_edges = np.arange(0.0 - wd_step / 2.0, 360.0, wd_step)

            # use wd_step to produce a wrapped version of wind_directions
            wind_directions_wrapped = self._wrap_wind_directions_near_360(
                self.wind_directions, wd_step
            )

            # Only keep the range with values in it
            wd_edges = wd_edges[wd_edges + wd_step > wind_directions_wrapped.min()]
            wd_edges = wd_edges[wd_edges - wd_step <= wind_directions_wrapped.max()]

        # Define the centers from the edges
        wd_centers = wd_edges[:-1] + wd_step / 2.0

        # Repeat for wind speeds
        if ws_edges is not None:
            ws_step = ws_edges[1] - ws_edges[0]

        else:
            ws_edges = np.arange(0.0 - ws_step / 2.0, 50.0, ws_step)

            # Only keep the range with values in it
            ws_edges = ws_edges[ws_edges + ws_step > self.wind_speeds.min()]
            ws_edges = ws_edges[ws_edges - ws_step <= self.wind_speeds.max()]

        # Define the centers from the edges
        ws_centers = ws_edges[:-1] + ws_step / 2.0

        # Now use pandas to get the tables need for wind rose
        df = pd.DataFrame(
            {
                "wd": wind_directions_wrapped,
                "ws": self.wind_speeds,
                "freq_val": np.ones(len(wind_directions_wrapped)),
            }
        )

        # If bin_weights are passed in, apply these to the frequency
        # this is mostly used when resampling the wind rose
        if bin_weights is not None:
            df = df.assign(freq_val=df["freq_val"] * bin_weights)

        # Add turbulence intensities to dataframe
        df = df.assign(turbulence_intensities=self.turbulence_intensities)

        # If values is not none, add to dataframe
        if self.values is not None:
            df = df.assign(values=self.values)

        # Bin wind speed and wind direction and then group things up
        df = (
            df.assign(
                wd_bin=pd.cut(
                    df.wd, bins=wd_edges, labels=wd_centers, right=False, include_lowest=True
                )
            )
            .assign(
                ws_bin=pd.cut(
                    df.ws, bins=ws_edges, labels=ws_centers, right=False, include_lowest=True
                )
            )
            .drop(["wd", "ws"], axis=1)
        )

        # Convert wd_bin and ws_bin to categoricals to ensure all combinations
        # are considered and then group
        wd_cat = CategoricalDtype(categories=wd_centers, ordered=True)
        ws_cat = CategoricalDtype(categories=ws_centers, ordered=True)

        df = (
            df.assign(wd_bin=df["wd_bin"].astype(wd_cat))
            .assign(ws_bin=df["ws_bin"].astype(ws_cat))
            .groupby(["wd_bin", "ws_bin"], observed=False)
            .agg(["sum", "mean"])
        )
        # Flatten and combine levels using an underscore
        df.columns = ["_".join(col) for col in df.columns]

        # Collect the frequency table and reshape
        freq_table = df["freq_val_sum"].values.copy()
        freq_table = freq_table / freq_table.sum()
        freq_table = freq_table.reshape((len(wd_centers), len(ws_centers)))

        # Compute the TI table
        ti_table = df["turbulence_intensities_mean"].values.copy()
        ti_table = ti_table.reshape((len(wd_centers), len(ws_centers)))

        # If values is not none, compute the table
        if self.values is not None:
            value_table = df["values_mean"].values.copy()
            value_table = value_table.reshape((len(wd_centers), len(ws_centers)))
        else:
            value_table = None

        # Return a WindRose
        return WindRose(
            wd_centers,
            ws_centers,
            ti_table,
            freq_table,
            value_table,
            self.heterogenous_inflow_config_by_wd,
        )

    def to_wind_ti_rose(
        self,
        wd_step=2.0,
        ws_step=1.0,
        ti_step=0.02,
        wd_edges=None,
        ws_edges=None,
        ti_edges=None,
        bin_weights=None,
    ):
        """
        Converts the TimeSeries data to a WindTIRose.

        Args:
            wd_step (float, optional): Step size for wind direction (default is 2.0).
            ws_step (float, optional): Step size for wind speed (default is 1.0).
            ti_step (float, optional): Step size for turbulence intensity (default is 0.02).
            wd_edges (NDArrayFloat, optional): Custom wind direction edges. Defaults to None.
            ws_edges (NDArrayFloat, optional): Custom wind speed edges. Defaults to None.
            ti_edges (NDArrayFloat, optional): Custom turbulence intensity
                edges. Defaults to None.
            bin_weights (NDArrayFloat, optional): Bin weights for resampling.  Note these
                are primarily used by the resample resample_wind_rose function.
                Defaults to None.

        Returns:
            WindRose: A WindTIRose object based on the TimeSeries data.

        Notes:
            - If `wd_edges` is defined, it uses it to produce the wind direction bin edges.
            - If `wd_edges` is not defined, it determines `wd_edges` from the step and data.
            - If `ws_edges` is defined, it uses it for wind speed edges.
            - If `ws_edges` is not defined, it determines `ws_edges` from the step and data.
            - If `ti_edges` is defined, it uses it for turbulence intensity edges.
            - If `ti_edges` is not defined, it determines `ti_edges` from the step and data.
        """

        # If wd_edges is defined, then use it to produce the bin centers
        if wd_edges is not None:
            wd_step = wd_edges[1] - wd_edges[0]

            # use wd_step to produce a wrapped version of wind_directions
            wind_directions_wrapped = self._wrap_wind_directions_near_360(
                self.wind_directions, wd_step
            )

        # Else, determine wd_edges from the step and data
        else:
            wd_edges = np.arange(0.0 - wd_step / 2.0, 360.0, wd_step)

            # use wd_step to produce a wrapped version of wind_directions
            wind_directions_wrapped = self._wrap_wind_directions_near_360(
                self.wind_directions, wd_step
            )

            # Only keep the range with values in it
            wd_edges = wd_edges[wd_edges + wd_step > wind_directions_wrapped.min()]
            wd_edges = wd_edges[wd_edges - wd_step <= wind_directions_wrapped.max()]

        # Define the centers from the edges
        wd_centers = wd_edges[:-1] + wd_step / 2.0

        # Repeat for wind speeds
        if ws_edges is not None:
            ws_step = ws_edges[1] - ws_edges[0]

        else:
            ws_edges = np.arange(0.0 - ws_step / 2.0, 50.0, ws_step)

            # Only keep the range with values in it
            ws_edges = ws_edges[ws_edges + ws_step > self.wind_speeds.min()]
            ws_edges = ws_edges[ws_edges - ws_step <= self.wind_speeds.max()]

        # Define the centers from the edges
        ws_centers = ws_edges[:-1] + ws_step / 2.0

        # Repeat for turbulence intensities
        if ti_edges is not None:
            ti_step = ti_edges[1] - ti_edges[0]

        else:
            ti_edges = np.arange(0.0 - ti_step / 2.0, 1.0, ti_step)

            # Only keep the range with values in it
            ti_edges = ti_edges[ti_edges + ti_step > self.turbulence_intensities.min()]
            ti_edges = ti_edges[ti_edges - ti_step <= self.turbulence_intensities.max()]

        # Define the centers from the edges
        ti_centers = ti_edges[:-1] + ti_step / 2.0

        # Now use pandas to get the tables need for wind rose
        df = pd.DataFrame(
            {
                "wd": wind_directions_wrapped,
                "ws": self.wind_speeds,
                "ti": self.turbulence_intensities,
                "freq_val": np.ones(len(wind_directions_wrapped)),
            }
        )

        # If bin_weights are passed in, apply these to the frequency
        # this is mostly used when resampling the wind rose
        if bin_weights is not None:
            df = df.assign(freq_val=df["freq_val"] * bin_weights)

        # If values is not none, add to dataframe
        if self.values is not None:
            df = df.assign(values=self.values)

        # Bin wind speed, wind direction, and turbulence intensity and then group things up
        df = (
            df.assign(
                wd_bin=pd.cut(
                    df.wd, bins=wd_edges, labels=wd_centers, right=False, include_lowest=True
                )
            )
            .assign(
                ws_bin=pd.cut(
                    df.ws, bins=ws_edges, labels=ws_centers, right=False, include_lowest=True
                )
            )
            .assign(
                ti_bin=pd.cut(
                    df.ti, bins=ti_edges, labels=ti_centers, right=False, include_lowest=True
                )
            )
            .drop(["wd", "ws", "ti"], axis=1)
        )

        # Convert wd_bin, ws_bin, and ti_bin to categoricals to ensure all
        # combinations are considered and then group
        wd_cat = CategoricalDtype(categories=wd_centers, ordered=True)
        ws_cat = CategoricalDtype(categories=ws_centers, ordered=True)
        ti_cat = CategoricalDtype(categories=ti_centers, ordered=True)

        df = (
            df.assign(wd_bin=df["wd_bin"].astype(wd_cat))
            .assign(ws_bin=df["ws_bin"].astype(ws_cat))
            .assign(ti_bin=df["ti_bin"].astype(ti_cat))
            .groupby(["wd_bin", "ws_bin", "ti_bin"], observed=False)
            .agg(["sum", "mean"])
        )
        # Flatten and combine levels using an underscore
        df.columns = ["_".join(col) for col in df.columns]

        # Collect the frequency table and reshape
        freq_table = df["freq_val_sum"].values.copy()
        freq_table = freq_table / freq_table.sum()
        freq_table = freq_table.reshape((len(wd_centers), len(ws_centers), len(ti_centers)))

        # If values is not none, compute the table
        if self.values is not None:
            value_table = df["values_mean"].values.copy()
            value_table = value_table.reshape((len(wd_centers), len(ws_centers), len(ti_centers)))
        else:
            value_table = None

        # Return a WindTIRose
        return WindTIRose(
            wd_centers,
            ws_centers,
            ti_centers,
            freq_table,
            value_table,
            self.heterogenous_inflow_config_by_wd,
        )
