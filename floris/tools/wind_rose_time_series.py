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

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype


class WindRose:
    """
    In FLORIS v4, the WindRose class is used to drive FLORIS and optimization
    operations in which the inflow is characterized by the frequency of
    binned wind speed, wind direction and turbulence intensity values

    """

    def __init__(
        self,
        wind_directions,
        wind_speeds,
        freq_table=None,
        ti_table=None,
        price_table=None,
    ):
        """
        TODO: Write this later
        """

        # Save the wind speeds and directions
        self.wind_directions = wind_directions
        self.wind_speeds = wind_speeds

        # Also save gridded versions
        self.wd_grid, self.ws_grid = np.meshgrid(
            self.wind_directions, self.wind_speeds, indexing="ij"
        )

        # Save flat versions of each as well
        self.wd_flat = self.wd_grid.flatten()
        self.ws_flat = self.ws_grid.flatten()

        # If freq_table is not None, confirm it has correct dimension,
        # otherwise initialze to uniform probability
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

        # Save a flatten version
        self.freq_table_flat = self.freq_table.flatten()

        # If TI table is not None, confirm dimension
        # otherwise leave it None
        if ti_table is not None:
            if not ti_table.shape[0] == len(wind_directions):
                raise ValueError("ti_table first dimension must equal len(wind_directions)")
            if not ti_table.shape[1] == len(wind_speeds):
                raise ValueError("ti_table second dimension must equal len(wind_speeds)")
            self.ti_table = ti_table
            self.ti_table_flat = self.ti_table.flatten()
        else:
            self.ti_table = None
            self.ti_table_flat = None

        # If price_table is not None, confirm it has correct dimension,
        # otherwise initialze to all ones
        if price_table is not None:
            if not price_table.shape[0] == len(wind_directions):
                raise ValueError("price_table first dimension must equal len(wind_directions)")
            if not price_table.shape[1] == len(wind_speeds):
                raise ValueError("price_table second dimension must equal len(wind_speeds)")
            self.price_table = price_table
        else:
            self.price_table = np.ones((len(wind_directions), len(wind_speeds)))
        # Save a flatten version
        self.price_table_flat = self.price_table.flatten()

    def _unpack(self):
        """
        Unpack the values in a form which is ready for FLORIS' reinitialize function
        """

        # The unpacked versions start as the flat version of each
        wind_directions_unpack = self.wd_flat.copy()
        wind_speeds_unpack = self.ws_flat.copy()
        freq_table_unpack = self.freq_table_flat.copy()

        # Get a mask of combinations that are more than 0 occurences
        self.unpack_mask = freq_table_unpack > 0.0

        # Now mask thes values to as to only compute values with occurence over 0
        wind_directions_unpack = wind_directions_unpack[self.unpack_mask]
        wind_speeds_unpack = wind_speeds_unpack[self.unpack_mask]
        freq_table_unpack = freq_table_unpack[self.unpack_mask]

        # Repeat for turbulence intensity if not none
        if self.ti_table_flat is not None:
            ti_table_unpack = self.ti_table_flat[self.unpack_mask]
        else:
            ti_table_unpack = None

        # Now get unpacked price table
        price_table_unpack = self.price_table_flat[self.unpack_mask]

        return (
            wind_directions_unpack,
            wind_speeds_unpack,
            freq_table_unpack,
            ti_table_unpack,
            price_table_unpack,
        )


class TimeSeries:
    """
    In FLORIS v4, the TimeSeries class is used to drive FLORIS and optimization
    operations in which the inflow is by a sequence of wind speed, wind directino
    and turbulence intensitity values

    """

    def __init__(
        self,
        wind_directions,
        wind_speeds,
        turbulence_intensity=None,
        prices=None,
    ):
        """
        TODO: Write this later
        """

        # Wind speeds and wind directions must be the same length
        if len(wind_directions) != len(wind_speeds):
            raise ValueError("wind_directions and wind_speeds must be the same length")

        self.wind_directions = wind_directions
        self.wind_speeds = wind_speeds
        self.turbulence_intensity = turbulence_intensity
        self.prices = prices

        # Record findex
        self.n_findex = len(self.wind_directions)

    def _unpack(self):
        """
        Unpack the time series data to floris' reinitialize function
        """
        return (
            self.wind_directions.copy(),
            self.wind_speeds.copy(),
            self.turbulence_intensity.copy(),
            self.prices.copy(),
        )

    def _wrap_wind_directions_near_360(self, wind_directions, wd_step):
        """
        use wd_step to produce a wrapped version of wind_directions
        where values that are between [360 - wd_step/2.0,360] get mapped
        to negative numbers for binning
        """
        wind_directions_wrapped = wind_directions.copy()
        mask = wind_directions_wrapped >= 360 - wd_step / 2.0
        wind_directions_wrapped[mask] = wind_directions_wrapped[mask] - 360.0
        return wind_directions_wrapped

    def to_wind_rose(self, wd_step=2.0, ws_step=1.0, wd_edges=None, ws_edges=None):
        """
        TODO: Write this later
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
            wd_edges = wd_edges[wd_edges + wd_step >= wind_directions_wrapped.min()]
            wd_edges = wd_edges[wd_edges - wd_step <= wind_directions_wrapped.max()]

        # Define the centers from the edges
        wd_centers = wd_edges[:-1] + wd_step / 2.0

        # Repeat for wind speeds
        if ws_edges is not None:
            ws_step = ws_edges[1] - ws_edges[0]

        else:
            ws_edges = np.arange(0.0 - ws_step / 2.0, 50.0, ws_step)

            # Only keep the range with values in it
            ws_edges = ws_edges[ws_edges + ws_step >= self.wind_speeds.min()]
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

        # If turbulence_intensity is not none, add to dataframe
        if self.turbulence_intensity is not None:
            df = df.assign(turbulence_intensity=self.turbulence_intensity)

        # If prices is not none, add to dataframe
        if self.prices is not None:
            df = df.assign(prices=self.prices)

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
            .groupby(["wd_bin", "ws_bin"])
            .agg([np.sum, np.mean])
        )
        # Flatten and combine levels using an underscore
        df.columns = ["_".join(col) for col in df.columns]

        # Collect the frequency table and reshape
        freq_table = df["freq_val_sum"].values.copy()
        freq_table = freq_table / freq_table.sum()
        freq_table = freq_table.reshape((len(wd_centers), len(ws_centers)))

        # If turbulence intensity is not none, compute the table
        if self.turbulence_intensity is not None:
            ti_table = df["turbulence_intensity_mean"].values.copy()
            ti_table = ti_table.reshape((len(wd_centers), len(ws_centers)))
        else:
            ti_table = None

        # If prices is not none, compute the table
        if self.prices is not None:
            price_table = df["prices_mean"].values.copy()
            price_table = price_table.reshape((len(wd_centers), len(ws_centers)))
        else:
            price_table = None

        # Return a WindRose
        return WindRose(wd_centers, ws_centers, freq_table, ti_table, price_table)
