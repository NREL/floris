# Copyright 2024 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

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
            _,
            ti_table_unpack,
            _,
        ) = self.unpack()

        return wind_directions_unpack, wind_speeds_unpack, ti_table_unpack

    def unpack_freq(self):
        """Unpack frequency weighting"""

        (
            _,
            _,
            freq_table_unpack,
            _,
            _,
        ) = self.unpack()

        return freq_table_unpack


class WindRose(WindDataBase):
    """
    In FLORIS v4, the WindRose class is used to drive FLORIS and optimization
    operations in which the inflow is characterized by the frequency of
    binned wind speed, wind direction and turbulence intensity values

    Args:
        wind_directions: NumPy array of wind directions (NDArrayFloat).
        wind_speeds: NumPy array of wind speeds (NDArrayFloat).
        freq_table: Frequency table for binned wind direction, wind speed
            values (NDArrayFloat, optional).   Must have dimension
            (n_wind_directions, n_wind_speeds).  Defaults to None in which case
            uniform frequency of all bins is assumed.
        ti_table: Turbulence intensity table for binned wind direction, wind
            speed values (NDArrayFloat, optional).   Must have dimension
            (n_wind_directions, n_wind_speeds). Defaults to None (no change to
            turbulence intensity)
        value_table: Value table for binned wind direction, wind
            speed values (NDArrayFloat, optional).  Must have dimension
            (n_wind_directions, n_wind_speeds).  Defaults to None in which case
            uniform values are assumed.  Value can be used to weight power in
            each bin to compute the total value of the energy produced
        compute_zero_freq_occurrence: Flag indicating whether to compute zero
            frequency occurrences (bool, optional).  Defaults to False.

    """

    def __init__(
        self,
        wind_directions: NDArrayFloat,
        wind_speeds: NDArrayFloat,
        freq_table: NDArrayFloat | None = None,
        ti_table: NDArrayFloat | None = None,
        value_table: NDArrayFloat | None = None,
        compute_zero_freq_occurrence: bool = False,
    ):
        if not isinstance(wind_directions, np.ndarray):
            raise TypeError("wind_directions must be a NumPy array")

        if not isinstance(wind_speeds, np.ndarray):
            raise TypeError("wind_speeds must be a NumPy array")

        # Save the wind speeds and directions
        self.wind_directions = wind_directions
        self.wind_speeds = wind_speeds

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

        # If TI table is not None, confirm dimension
        # otherwise leave it None
        if ti_table is not None:
            if not ti_table.shape[0] == len(wind_directions):
                raise ValueError("ti_table first dimension must equal len(wind_directions)")
            if not ti_table.shape[1] == len(wind_speeds):
                raise ValueError("ti_table second dimension must equal len(wind_speeds)")
        self.ti_table = ti_table

        # If value_table is not None, confirm it has correct dimension,
        # otherwise initialize to all ones
        if value_table is not None:
            if not value_table.shape[0] == len(wind_directions):
                raise ValueError("value_table first dimension must equal len(wind_directions)")
            if not value_table.shape[1] == len(wind_speeds):
                raise ValueError("value_table second dimension must equal len(wind_speeds)")
        self.value_table = value_table

        # Save whether zero occurrence cases should be computed
        self.compute_zero_freq_occurrence = compute_zero_freq_occurrence

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

        # TI table
        if self.ti_table is not None:
            self.ti_table_flat = self.ti_table.flatten()
        else:
            self.ti_table_flat = None

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

        # Now mask thes values according to self.non_zero_freq_mask
        wind_directions_unpack = wind_directions_unpack[self.non_zero_freq_mask]
        wind_speeds_unpack = wind_speeds_unpack[self.non_zero_freq_mask]
        freq_table_unpack = freq_table_unpack[self.non_zero_freq_mask]

        # Repeat for turbulence intensity if not none
        if self.ti_table_flat is not None:
            ti_table_unpack = self.ti_table_flat[self.non_zero_freq_mask].copy()
        else:
            ti_table_unpack = None

        # Now get unpacked value table
        if self.value_table_flat is not None:
            value_table_unpack = self.value_table_flat[self.non_zero_freq_mask].copy()
        else:
            value_table_unpack = None

        return (
            wind_directions_unpack,
            wind_speeds_unpack,
            freq_table_unpack,
            ti_table_unpack,
            value_table_unpack,
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
            self.wd_flat, self.ws_flat, self.ti_table_flat, self.value_table_flat
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

        ax.plot(self.ws_flat, self.ti_table_flat*100, marker=marker, ls=ls, color=color)
        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Turbulence Intensity (%)")
        ax.grid(True)


class TimeSeries(WindDataBase):
    """
    In FLORIS v4, the TimeSeries class is used to drive FLORIS and optimization
    operations in which the inflow is by a sequence of wind direction, wind speed
    and turbulence intensity values

    Args:
        wind_directions: NumPy array of wind directions (NDArrayFloat).
        wind_speeds: NumPy array of wind speeds (NDArrayFloat).
        turbulence_intensities:  NumPy array of wind speeds (NDArrayFloat, optional).
            Defaults to None
        values:  NumPy array of electricity values (NDArrayFloat, optional).
            Defaults to None

    """

    def __init__(
        self,
        wind_directions: NDArrayFloat,
        wind_speeds: NDArrayFloat,
        turbulence_intensities: NDArrayFloat | None = None,
        values: NDArrayFloat | None = None,
    ):
        # Wind speeds and wind directions must be the same length
        if len(wind_directions) != len(wind_speeds):
            raise ValueError("wind_directions and wind_speeds must be the same length")

        # If turbulence_intensities is not None, must be same length as wind_directions
        if turbulence_intensities is not None:
            if len(wind_directions) != len(turbulence_intensities):
                raise ValueError(
                    "wind_directions and turbulence_intensities must be the same length"
                )

        # If values is not None, must be same length as wind_directions
        if values is not None:
            if len(wind_directions) != len(values):
                raise ValueError("wind_directions and values must be the same length")

        self.wind_directions = wind_directions
        self.wind_speeds = wind_speeds
        self.turbulence_intensities = turbulence_intensities
        self.values = values

        # Record findex
        self.n_findex = len(self.wind_directions)

    def unpack(self):
        """
        Unpack the time series data in a manner consistent with wind rose unpack
        """

        # to match wind_rose, make a uniform frequency
        uniform_frequency = np.ones_like(self.wind_directions)
        uniform_frequency = uniform_frequency / uniform_frequency.sum()

        return (
            self.wind_directions,
            self.wind_speeds,
            uniform_frequency,
            self.turbulence_intensities,
            self.values,
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

        # If turbulence_intensities is not none, add to dataframe
        if self.turbulence_intensities is not None:
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

        # If turbulence intensity is not none, compute the table
        if self.turbulence_intensities is not None:
            ti_table = df["turbulence_intensities_mean"].values.copy()
            ti_table = ti_table.reshape((len(wd_centers), len(ws_centers)))
        else:
            ti_table = None

        # If values is not none, compute the table
        if self.values is not None:
            value_table = df["values_mean"].values.copy()
            value_table = value_table.reshape((len(wd_centers), len(ws_centers)))
        else:
            value_table = None

        # Return a WindRose
        return WindRose(wd_centers, ws_centers, freq_table, ti_table, value_table)
