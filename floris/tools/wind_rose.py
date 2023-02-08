# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

# TODO
# 1: reorganize into private and public methods
# 2: Include smoothing?

import os
import pickle

import dateutil
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

import floris.utilities as geo


# from pyproj import Proj



class WindRose:
    """
    The WindRose class is used to organize information about the frequency of
    occurance of different combinations of wind speed and wind direction (and
    other optimal wind variables). A WindRose object can be used to help
    calculate annual energy production (AEP) when combined with Floris power
    calculations for different wind conditions. Several methods exist for
    populating a WindRose object with wind data. WindRose also contains methods
    for visualizing wind roses.

    References:
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: wr-
    """

    def __init__(self,):
        """
        Instantiate a WindRose object and set some initial parameter values.
        No explicit arguments required, and an additional method will need to
        be called to populate the WindRose object with data.
        """
        # Initialize some varibles
        self.num_wd = 0
        self.num_ws = 0
        self.wd_step = 1.0
        self.ws_step = 5.0
        self.wd = np.array([])
        self.ws = np.array([])
        self.df = pd.DataFrame()

    def save(self, filename):
        """
        This method saves the WindRose data as a pickle file so that it can be
        imported into a WindRose object later.

        Args:
            filename (str): Path and filename of pickle file to save.
        """
        pickle.dump(
            [
                self.num_wd,
                self.num_ws,
                self.wd_step,
                self.ws_step,
                self.wd,
                self.ws,
                self.df,
            ],
            open(filename, "wb"),
        )

    def load(self, filename):
        """
        This method loads data from a previously saved WindRose pickle file
        into a WindRose object.

        Args:
            filename (str): Path and filename of pickle file to load.

        Returns:
            int, int, float, float, np.array, np.array, pandas.DataFrame:

                -   Number of wind direction bins.
                -   Number of wind speed bins.
                -   Wind direction bin size (deg).
                -   Wind speed bin size (m/s).
                -   List of wind direction bin center values (deg).
                -   List of wind speed bin center values (m/s).
                -   DataFrame containing at least the following columns:

                    - **wd** (*float*) - Wind direction bin center values (deg).
                    - **ws** (*float*) - Wind speed bin center values (m/s).
                    - **freq_val** (*float*) - The frequency of occurance of
                      the wind conditions in the other columns.
        """
        (
            self.num_wd,
            self.num_ws,
            self.wd_step,
            self.ws_step,
            self.wd,
            self.ws,
            self.df,
        ) = pickle.load(open(filename, "rb"))

        return self.df

    def resample_wind_speed(self, df, ws=np.arange(0, 26, 1.0)):
        """
        This method resamples the wind speed bins using the specified wind
        speed bin center values. The frequency values are adjusted accordingly.

        Args:
            df (pandas.DataFrame): Wind rose DataFrame containing at least the
                following columns:

                - **wd** (*float*) - Wind direction bin center values (deg).
                - **ws** (*float*) - Wind speed bin center values (m/s).
                - **freq_val** (*float*) - The frequency of occurance of the
                  wind conditions in the other columns.

            ws (np.array, optional): List of new wind speed center bins (m/s).
                Defaults to np.arange(0, 26, 1.).

        Returns:
            pandas.DataFrame: Wind rose DataFrame with the resampled wind speed
            bins and frequencies containing at least the following columns:

                - **wd** (*float*) - Wind direction bin center values (deg).
                - **ws** (*float*) - New wind speed bin center values (m/s).
                - **freq_val** (*float*) - The frequency of occurance of the
                  new wind conditions in the other columns.
        """
        # Make a copy of incoming dataframe
        df = df.copy(deep=True)

        # Get the wind step
        ws_step = ws[1] - ws[0]

        # Ws
        ws_edges = ws - ws_step / 2.0
        ws_edges = np.append(ws_edges, np.array(ws[-1] + ws_step / 2.0))

        # Cut wind speed onto bins
        df["ws"] = pd.cut(df.ws, ws_edges, labels=ws)

        # Regroup
        df = df.groupby([c for c in df.columns if c != "freq_val"]).sum()

        # Fill nans
        df = df.fillna(0)

        # Reset the index
        df = df.reset_index()

        # Set to float
        for c in [c for c in df.columns if c != "freq_val"]:
            df[c] = df[c].astype(float)
            df[c] = df[c].astype(float)

        return df

    def internal_resample_wind_speed(self, ws=np.arange(0, 26, 1.0)):
        """
        Internal method for resampling wind speed into desired bins. The
        frequency values are adjusted accordingly. Modifies data within
        WindRose object without explicit return.

        TODO: make a private method

        Args:
            ws (np.array, optional): Vector of wind speed bin centers for
                the wind rose (m/s). Defaults to np.arange(0, 26, 1.).
        """
        # Update ws and wd binning
        self.ws = ws
        self.num_ws = len(ws)
        self.ws_step = ws[1] - ws[0]

        # Update internal data frame
        self.df = self.resample_wind_speed(self.df, ws)

    def resample_wind_direction(self, df, wd=np.arange(0, 360, 5.0)):
        """
        This method resamples the wind direction bins using the specified wind
        direction bin center values. The frequency values are adjusted
        accordingly.

        Args:
            df (pandas.DataFrame): Wind rose DataFrame containing at least the
                following columns:

                - **wd** (*float*) - Wind direction bin center values (deg).
                - **ws** (*float*) - Wind speed bin center values (m/s).
                - **freq_val** (*float*) - The frequency of occurance of the
                  wind conditions in the other columns.

            wd (np.array, optional): List of new wind direction center bins
                (deg). Defaults to np.arange(0, 360, 5.).

        Returns:
            pandas.DataFrame: Wind rose DataFrame with the resampled wind
            direction bins and frequencies containing at least the following
            columns:

                - **wd** (*float*) - New wind direction bin center values (deg).
                - **ws** (*float*) - Wind speed bin center values (m/s).
                - **freq_val** (*float*) - The frequency of occurance of the
                  new wind conditions in the other columns.
        """
        # Make a copy of incoming dataframe
        df = df.copy(deep=True)

        # Get the wind step
        wd_step = wd[1] - wd[0]

        # Get bin edges
        wd_edges = wd - wd_step / 2.0
        wd_edges = np.append(wd_edges, np.array(wd[-1] + wd_step / 2.0))

        # Get the overhangs
        negative_overhang = wd_edges[0]
        positive_overhang = wd_edges[-1] - 360.0

        # Need potentially to wrap high angle direction to negative for correct
        # binning
        df["wd"] = geo.wrap_360(df.wd)
        if negative_overhang < 0:
            print("Correcting negative Overhang:%.1f" % negative_overhang)
            df["wd"] = np.where(
                df.wd.values >= 360.0 + negative_overhang,
                df.wd.values - 360.0,
                df.wd.values,
            )

        # Check on other side
        if positive_overhang > 0:
            print("Correcting positive Overhang:%.1f" % positive_overhang)
            df["wd"] = np.where(
                df.wd.values <= positive_overhang, df.wd.values + 360.0, df.wd.values
            )

        # Cut into bins
        df["wd"] = pd.cut(df.wd, wd_edges, labels=wd)

        # Regroup
        df = df.groupby([c for c in df.columns if c != "freq_val"]).sum()

        # Fill nans
        df = df.fillna(0)

        # Reset the index
        df = df.reset_index()

        # Set to float Re-wrap
        for c in [c for c in df.columns if c != "freq_val"]:
            df[c] = df[c].astype(float)
            df[c] = df[c].astype(float)
        df["wd"] = geo.wrap_360(df.wd)

        return df

    def internal_resample_wind_direction(self, wd=np.arange(0, 360, 5.0)):
        """
        Internal method for resampling wind direction into desired bins. The
        frequency values are adjusted accordingly. Modifies data within
        WindRose object without explicit return.

        TODO: make a private method

        Args:
            wd (np.array, optional): Vector of wind direction bin centers for
                the wind rose (deg). Defaults to np.arange(0, 360, 5.).
        """
        # Update ws and wd binning
        self.wd = wd
        self.num_wd = len(wd)
        self.wd_step = wd[1] - wd[0]

        # Update internal data frame
        self.df = self.resample_wind_direction(self.df, wd)

    def resample_column(self, df, col, bins):
        """
        This method resamples the specified wind parameter column using the
        specified bin center values. The frequency values are adjusted
        accordingly.

        Args:
            df (pandas.DataFrame): Wind rose DataFrame containing at least the
                following columns as well as *col*:

                - **wd** (*float*) - Wind direction bin center values (deg).
                - **ws** (*float*) - Wind speed bin center values (m/s).
                - **freq_val** (*float*) - The frequency of occurance of the
                  wind conditions in the other columns.

            col (str): The name of the column to resample.
            bins (np.array): List of new bin center values for the specified
                column.

        Returns:
            pandas.DataFrame: Wind rose DataFrame with the resampled wind
            parameter bins and frequencies containing at least the following
            columns as well as *col*:

                - **wd** (*float*) - Wind direction bin center values (deg).
                - **ws** (*float*) - Wind speed bin center values (m/s).
                - **freq_val** (*float*) - The frequency of occurance of the
                  new wind conditions in the other columns.
        """
        # Make a copy of incoming dataframe
        df = df.copy(deep=True)

        # Cut into bins, make first and last bins extend to -/+ infinity
        var_edges = np.append(0.5 * (bins[1:] + bins[:-1]), np.inf)
        var_edges = np.append(-np.inf, var_edges)
        df[col] = pd.cut(df[col], var_edges, labels=bins)

        # Regroup
        df = df.groupby([c for c in df.columns if c != "freq_val"]).sum()

        # Fill nans
        df = df.fillna(0)

        # Reset the index
        df = df.reset_index()

        # Set to float
        for c in [c for c in df.columns if c != "freq_val"]:
            df[c] = df[c].astype(float)

        return df

    def internal_resample_column(self, col, bins):
        """
        Internal method for resampling column into desired bins. The frequency
        values are adjusted accordingly. Modifies data within WindRose object
        without explicit return.

        TODO: make a private method

        Args:
            col (str): Name of column to resample.
            bins (np.array): Vector of bins for the WindRose column.
        """
        # Update internal data frame
        self.df = self.resample_column(self.df, col, bins)

    def resample_average_ws_by_wd(self, df):
        """
        This method calculates the mean wind speed for each wind direction bin
        and resamples the wind rose, resulting in a single mean wind speed per
        wind direction bin. The frequency values are adjusted accordingly.

        Args:
            df (pandas.DataFrame): Wind rose DataFrame containing at least the
                following columns:

                - **wd** (*float*) - Wind direction bin center values (deg).
                - **ws** (*float*) - Wind speed bin center values (m/s).
                - **freq_val** (*float*) - The frequency of occurance of the
                  wind conditions in the other columns.

        Returns:
            pandas.DataFrame: Wind rose DataFrame with the resampled wind speed
            bins and frequencies containing at least the following columns:

                - **wd** (*float*) - Wind direction bin center values (deg).
                - **ws** (*float*) - The average wind speed for each wind
                  direction bin (m/s).
                - **freq_val** (*float*) - The frequency of occurance of the
                  new wind conditions in the other columns.
        """
        # Make a copy of incoming dataframe
        df = df.copy(deep=True)

        ws_avg = []

        for val in df.wd.unique():
            ws_avg.append(
                np.array(
                    df.loc[df["wd"] == val]["ws"] * df.loc[df["wd"] == val]["freq_val"]
                ).sum()
                / df.loc[df["wd"] == val]["freq_val"].sum()
            )

        # Regroup
        df = df.groupby("wd").sum()

        df["ws"] = ws_avg

        # Reset the index
        df = df.reset_index()

        # Set to float
        df["ws"] = df.ws.astype(float)
        df["wd"] = df.wd.astype(float)

        return df

    def internal_resample_average_ws_by_wd(self, wd=np.arange(0, 360, 5.0)):
        """
        This internal method calculates the mean wind speed for each specified
        wind direction bin and resamples the wind rose, resulting in a single
        mean wind speed per wind direction bin. The frequency values are
        adjusted accordingly.

        TODO: make an internal method

        Args:
            wd (np.arange, optional): Wind direction bin centers (deg).
            Defaults to np.arange(0, 360, 5.).

        Returns:
            pandas.DataFrame: Wind rose DataFrame with the resampled wind speed
            bins and frequencies containing at least the following columns:

                - **wd** (*float*) - Wind direction bin center values (deg).
                - **ws** (*float*) - The average wind speed for each wind
                  direction bin (m/s).
                - **freq_val** (*float*) - The frequency of occurance of the
                  new wind conditions in the other columns.
        """
        # Update ws and wd binning
        self.wd = wd
        self.num_wd = len(wd)
        self.wd_step = wd[1] - wd[0]

        # Update internal data frame
        self.df = self.resample_average_ws_by_wd(self.df)

    def interpolate(
        self,
        wind_directions: np.ndarray,
        wind_speeds: np.ndarray,
        mirror_0_to_360=True,
        fill_value=0.0,
        method="linear"
    ):
        """
        This method returns a linear interpolant that will return the occurrence
        frequency for any given wind direction and wind speed combination(s).
        This can be particularly useful when evaluating the wind rose at a
        higher frequency than the input data is provided.

        Args:
            wind_directions (np.ndarray): One or multi-dimensional array containing
            the wind direction values at which the wind rose frequency of occurrence
            should be evaluated.
            wind_speeds (np.ndarray): One or multi-dimensional array containing
            the wind speed values at which the wind rose frequency of occurrence
            should be evaluated.
            mirror_0_to_360 (bool, optional): This function copies the wind rose
            frequency values from 0 deg to 360 deg. This can be useful when, for example,
            the wind rose is only calculated until 357 deg but then interpolant is
            requesting values at 359 deg. Defaults to True.
            fill_value (float, optional): Fill value for the interpolant when
            interpolating values outside of the data region. Defaults to 0.0.
            method (str, optional): The interpolation method. Options are 'linear' and
            'nearest'. Recommended usage is 'linear'. Defaults to 'linear'.

        Returns:
            scipy.interpolate.LinearNDInterpolant: Linear interpolant for the
            wind rose currently available in the class (self.df).

        Example:
            wr = wind_rose.WindRose()
            wr.make_wind_rose_from_user_data(...)
            freq_floris = wr.interpolate(floris_wind_direction_grid, floris_wind_speed_grid)
        """
        if method == "linear":
            interpolator = LinearNDInterpolator
        elif method == "nearest":
            interpolator = NearestNDInterpolator
        else:
            UserWarning("Unknown interpolation method: '{:s}'".format(method))

        # Load windrose information from self
        df = self.df.copy()

        if mirror_0_to_360:
            # Copy values from 0 deg over to 360 deg
            df_copy = df[df["wd"] == 0.0].copy()
            df_copy["wd"] = 360.0
            df = pd.concat([df, df_copy], axis=0)

        interp = interpolator(
            points=df[["wd", "ws"]],
            values=df["freq_val"],
            fill_value=fill_value
        )
        return interp(wind_directions, wind_speeds)

    def weibull(self, x, k=2.5, lam=8.0):
        """
        This method returns a Weibull distribution corresponding to the input
        data array (typically wind speed) using the specified Weibull
        parameters.

        Args:
            x (np.array): List of input data (typically binned wind speed
                observations).
            k (float, optional): Weibull shape parameter. Defaults to 2.5.
            lam (float, optional): Weibull scale parameter. Defaults to 8.0.

        Returns:
            np.array: Weibull distribution probabilities corresponding to
            values in the input array.
        """
        return (k / lam) * (x / lam) ** (k - 1) * np.exp(-((x / lam) ** k))

    def make_wind_rose_from_weibull(
        self, wd=np.arange(0, 360, 5.0), ws=np.arange(0, 26, 1.0)
    ):
        """
        Populate WindRose object with an example wind rose with wind speed
        frequencies given by a Weibull distribution. The wind direction
        frequencies are initialized according to an example distribution.

        Args:
            wd (np.array, optional): Wind direciton bin centers (deg). Defaults
            to np.arange(0, 360, 5.).
            ws (np.array, optional): Wind speed bin centers (m/s). Defaults to
                np.arange(0, 26, 1.).

        Returns:
            pandas.DataFrame: Wind rose DataFrame containing at least the
            following columns:

                - **wd** (*float*) - Wind direction bin center values (deg).
                - **ws** (*float*) - Wind speed bin center values (m/s).
                - **freq_val** (*float*) - The frequency of occurance of the
                  wind conditions in the other columns.
        """
        # Use an assumed wind-direction for dir frequency
        wind_dir = [
            0,
            22.5,
            45,
            67.5,
            90,
            112.5,
            135,
            157.5,
            180,
            202.5,
            225,
            247.5,
            270,
            292.5,
            315,
            337.5,
        ]
        freq_dir = [
            0.064,
            0.04,
            0.038,
            0.036,
            0.045,
            0.05,
            0.07,
            0.08,
            0.11,
            0.08,
            0.05,
            0.036,
            0.048,
            0.058,
            0.095,
            0.10,
        ]

        freq_wd = np.interp(wd, wind_dir, freq_dir)
        freq_ws = self.weibull(ws)

        freq_tot = np.zeros(len(wd) * len(ws))
        wd_tot = np.zeros(len(wd) * len(ws))
        ws_tot = np.zeros(len(wd) * len(ws))

        count = 0
        for i in range(len(wd)):
            for j in range(len(ws)):
                wd_tot[count] = wd[i]
                ws_tot[count] = ws[j]

                freq_tot[count] = freq_wd[i] * freq_ws[j]
                count = count + 1

        # renormalize
        freq_tot = freq_tot / np.sum(freq_tot)

        # Load the wind toolkit data into a dataframe
        df = pd.DataFrame()

        # Start by simply round and wrapping the wind direction and wind speed
        # columns
        df["wd"] = wd_tot
        df["ws"] = ws_tot

        # Now group up
        df["freq_val"] = freq_tot

        # Save the df at this point
        self.df = df
        # TODO is there a reason self.df is updated AND returned?
        return self.df

    def make_wind_rose_from_user_data(
        self, wd_raw, ws_raw, *args, wd=np.arange(0, 360, 5.0), ws=np.arange(0, 26, 1.0)
    ):
        """
        This method populates the WindRose object given user-specified
        observations of wind direction, wind speed, and additional optional
        variables. The wind parameters are binned and the frequencies of
        occurance of each binned wind condition combination are calculated.

        Args:
            wd_raw (array-like): An array-like list of all wind direction
                observations used to calculate the normalized frequencies (deg).
            ws_raw (array-like): An array-like list of all wind speed
                observations used to calculate the normalized frequencies (m/s).
            *args: Variable length argument list consisting of a sequence of
                the following alternating arguments:

                -   string - Name of additional wind parameters to include in
                    wind rose.
                -   array-like - Values of the additional wind parameters used
                    to calculate the frequencies of occurance
                -   np.array - Bin center values for binning the additional
                    wind parameters.

            wd (np.array, optional): Wind direction bin centers (deg). Defaults
                to np.arange(0, 360, 5.).
            ws (np.array, optional): Wind speed bin limits (m/s). Defaults to
                np.arange(0, 26, 1.).

        Returns:
            pandas.DataFrame: Wind rose DataFrame containing at least the
            following columns:

                - **wd** (*float*) - Wind direction bin center values (deg).
                - **ws** (*float*) - Wind speed bin center values (m/s).
                - **freq_val** (*float*) - The frequency of occurance of the
                  wind conditions in the other columns.
        """
        df = pd.DataFrame()

        # convert inputs to np.array
        wd_raw = np.array(wd_raw)
        ws_raw = np.array(ws_raw)

        # Start by simply round and wrapping the wind direction and wind speed
        # columns
        df["wd"] = geo.wrap_360(wd_raw.round())
        df["ws"] = ws_raw.round()

        # Loop through *args and assign new dataframe columns after cutting
        # into possibly irregularly-spaced bins
        for in_var in range(0, len(args), 3):
            df[args[in_var]] = np.array(args[in_var + 1])

            # Cut into bins, make first and last bins extend to -/+ infinity
            var_edges = np.append(
                0.5 * (args[in_var + 2][1:] + args[in_var + 2][:-1]), np.inf
            )
            var_edges = np.append(-np.inf, var_edges)
            df[args[in_var]] = pd.cut(
                df[args[in_var]], var_edges, labels=args[in_var + 2]
            )

        # Now group up
        df["freq_val"] = 1.0
        df = df.groupby([c for c in df.columns if c != "freq_val"]).sum()
        df["freq_val"] = df.freq_val.astype(float) / df.freq_val.sum()
        df = df.reset_index()

        # Save the df at this point
        self.df = df

        # Resample onto the provided wind speed and wind direction binnings
        self.internal_resample_wind_speed(ws=ws)
        self.internal_resample_wind_direction(wd=wd)

        return self.df

    def read_wind_rose_csv(
        self,
        filename
    ):

        #Read in the csv
        self.df = pd.read_csv(filename)

        # Renormalize the frequency column
        self.df["freq_val"] = self.df["freq_val"] / self.df["freq_val"].sum()

        # Call the resample function in order to set all the internal variables
        self.internal_resample_wind_speed(ws=self.df.ws.unique())
        self.internal_resample_wind_direction(wd=self.df.wd.unique())


    def make_wind_rose_from_user_dist(
        self,
        wd_raw,
        ws_raw,
        freq_val,
        *args,
        wd=np.arange(0, 360, 5.0),
        ws=np.arange(0, 26, 1.0),
    ):
        """
        This method populates the WindRose object given user-specified
        combinations of wind direction, wind speed, additional optional
        variables, and the corresponding frequencies of occurance. The wind
        parameters are binned using the specified wind parameter bin center
        values and the corresponding frequencies of occrance are calculated.

        Args:
            wd_raw (array-like): An array-like list of wind directions
                corresponding to the specified frequencies of occurance (deg).
            wd_raw (array-like): An array-like list of wind speeds
                corresponding to the specified frequencies of occurance (m/s).
            freq_val (array-like): An array-like list of normalized frequencies
                corresponding to the provided wind parameter combinations.
            *args: Variable length argument list consisting of a sequence of
                the following alternating arguments:

                -   string - Name of additional wind parameters to include in
                    wind rose.
                -   array-like - Values of the additional wind parameters
                    corresponding to the specified frequencies of occurance.
                -   np.array - Bin center values for binning the additional
                    wind parameters.

            wd (np.array, optional): Wind direction bin centers (deg). Defaults
                to np.arange(0, 360, 5.).
            ws (np.array, optional): Wind speed bin centers (m/s). Defaults to
                np.arange(0, 26, 1.).

        Returns:
            pandas.DataFrame: Wind rose DataFrame containing at least the
            following columns:

                - **wd** (*float*) - Wind direction bin center values (deg).
                - **ws** (*float*) - Wind speed bin center values (m/s).
                - **freq_val** (*float*) - The frequency of occurance of the
                  wind conditions in the other columns.
        """
        df = pd.DataFrame()

        # convert inputs to np.array
        wd_raw = np.array(wd_raw)
        ws_raw = np.array(ws_raw)

        # Start by simply wrapping the wind direction column
        df["wd"] = geo.wrap_360(wd_raw)
        df["ws"] = ws_raw

        # Loop through *args and assign new dataframe columns
        for in_var in range(0, len(args), 3):
            df[args[in_var]] = np.array(args[in_var + 1])

        # Assign frequency column
        df["freq_val"] = np.array(freq_val)
        df["freq_val"] = df["freq_val"] / df["freq_val"].sum()

        # Save the df at this point
        self.df = df

        # Resample onto the provided wind variable binnings
        self.internal_resample_wind_speed(ws=ws)
        self.internal_resample_wind_direction(wd=wd)

        # Loop through *args and resample using provided binnings
        for in_var in range(0, len(args), 3):
            self.internal_resample_column(args[in_var], args[in_var + 2])

        return self.df

    def parse_wind_toolkit_folder(
        self,
        folder_name,
        wd=np.arange(0, 360, 5.0),
        ws=np.arange(0, 26, 1.0),
        limit_month=None,
    ):
        """
        This method populates the WindRose object given raw wind direction and
        wind speed data saved in csv files downloaded from the WIND Toolkit
        application (see https://www.nrel.gov/grid/wind-toolkit.html for more
        information). The wind parameters are binned using the specified wind
        parameter bin center values and the corresponding frequencies of
        occurance are calculated.

        Args:
            folder_name (str): Path to the folder containing the WIND Toolkit
                data files.
            wd (np.array, optional): Wind direction bin centers (deg). Defaults
                to np.arange(0, 360, 5.).
            ws (np.array, optional): Wind speed bin centers (m/s). Defaults to
                np.arange(0, 26, 1.).
            limit_month (list, optional): List of ints of month(s) (e.g., 1, 2
                3...) to consider when calculating the wind condition
                frequencies. If none are specified, all months will be used.
                Defaults to None.

        Returns:
            pandas.DataFrame: Wind rose DataFrame containing the following
            columns:

                - **wd** (*float*) - Wind direction bin center values (deg).
                - **ws** (*float*) - Wind speed bin center values (m/s).
                - **freq_val** (*float*) - The frequency of occurance of the
                  wind conditions in the other columns.
        """
        # Load the wind toolkit data into a dataframe
        df = self.load_wind_toolkit_folder(folder_name, limit_month=limit_month)

        # Start by simply round and wrapping the wind direction and wind speed
        # columns
        df["wd"] = geo.wrap_360(df.wd.round())
        df["ws"] = geo.wrap_360(df.ws.round())

        # Now group up
        df["freq_val"] = 1.0
        df = df.groupby(["ws", "wd"]).sum()
        df["freq_val"] = df.freq_val.astype(float) / df.freq_val.sum()
        df = df.reset_index()

        # Save the df at this point
        self.df = df

        # Resample onto the provided wind speed and wind direction binnings
        self.internal_resample_wind_speed(ws=ws)
        self.internal_resample_wind_direction(wd=wd)

        return self.df

    def load_wind_toolkit_folder(self, folder_name, limit_month=None):
        """
        This method imports raw wind direction and wind speed data saved in csv
        files in the specified folder downloaded from the WIND Toolkit
        application (see https://www.nrel.gov/grid/wind-toolkit.html for more
        information).

        TODO: make private method?

        Args:
            folder_name (str): Path to the folder containing the WIND Toolkit
                csv data files.
            limit_month (list, optional): List of ints of month(s) (e.g., 1, 2,
                3...) to consider when calculating the wind condition
                frequencies. If none are specified, all months will be used.
                Defaults to None.

        Returns:
            pandas.DataFrame: DataFrame containing the following columns:

                - **wd** (*float*) - Raw wind direction data (deg).
                - **ws** (*float*) - Raw wind speed data (m/s).
        """
        file_list = os.listdir(folder_name)
        file_list = [os.path.join(folder_name, f) for f in file_list if ".csv" in f]

        df = pd.DataFrame()
        for f_idx, f in enumerate(file_list):
            print("%d of %d: %s" % (f_idx, len(file_list), f))
            df_temp = self.load_wind_toolkit_file(f, limit_month=limit_month)
            df = df.append(df_temp)

        return df

    def load_wind_toolkit_file(self, filename, limit_month=None):
        """
        This method imports raw wind direction and wind speed data saved in the
        specified csv file downloaded from the WIND Toolkit application (see
        https://www.nrel.gov/grid/wind-toolkit.html for more information).

        TODO: make private method?

        Args:
            filename (str): Path to the WIND Toolkit csv file.
            limit_month (list, optional): List of ints of month(s) (e.g., 1, 2,
                3...) to consider when calculating the wind condition
                frequencies. If none are specified, all months will be used.
                Defaults to None.

        Returns:
            pandas.DataFrame: DataFrame containing the following columns with
            data from the WIND Toolkit file:

                - **wd** (*float*) - Raw wind direction data (deg).
                - **ws** (*float*) - Raw wind speed data (m/s).
        """
        df = pd.read_csv(filename, header=3, sep=",")

        # If asked to limit to particular months
        if limit_month is not None:
            df = df[df.Month.isin(limit_month)]

        # Save just what I want
        speed_column = [c for c in df.columns if "speed" in c][0]
        direction_column = [c for c in df.columns if "direction" in c][0]
        df = df.rename(index=str, columns={speed_column: "ws", direction_column: "wd"})[
            ["wd", "ws"]
        ]

        return df

    def import_from_wind_toolkit_hsds(
        self,
        lat,
        lon,
        ht=100,
        wd=np.arange(0, 360, 5.0),
        ws=np.arange(0, 26, 1.0),
        include_ti=False,
        limit_month=None,
        limit_hour=None,
        st_date=None,
        en_date=None,
    ):
        """
        This method populates the WindRose object using wind data from the WIND
        Toolkit dataset (https://www.nrel.gov/grid/wind-toolkit.html) for the
        specified lat/long coordinate in the continental US. The wind data
        are obtained from the WIND Toolkit dataset using the HSDS service (see
        https://github.com/NREL/hsds-examples). The wind data returned is
        obtained from the nearest 2km x 2km grid point to the input
        coordinate and is limited to the years 2007-2013. The wind parameters
        are binned using the specified wind parameter bin center values and the
        corresponding frequencies of occrance are calculated.

        Requires h5pyd package, which can be installed using:
            pip install --user git+http://github.com/HDFGroup/h5pyd.git

        Then, make a configuration file at ~/.hscfg containing:

            hs_endpoint = https://developer.nrel.gov/api/hsds

            hs_username = None

            hs_password = None

            hs_api_key = 3K3JQbjZmWctY0xmIfSYvYgtIcM3CN0cb1Y2w9bf

        The example API key above is for demonstation and is
        rate-limited per IP. To get your own API key, visit
        https://developer.nrel.gov/signup/.

        More information can be found at: https://github.com/NREL/hsds-examples.

        Args:
            lat (float): Latitude in degrees.
            lon (float): Longitude in degrees.
            ht (int, optional): The height above ground where wind
                information is obtained (m). Defaults to 100.
            wd (np.array, optional): Wind direction bin centers (deg). Defaults
                to np.arange(0, 360, 5.).
            ws (np.array, optional): Wind speed bin centers (m/s). Defaults to
                np.arange(0, 26, 1.).
            include_ti (bool, optional): Determines whether turbulence
                intensity is included as an additional parameter. If True, TI
                is added as an additional wind rose variable, estimated based
                on the Obukhov length from WIND Toolkit. Defaults to False.
            limit_month (list, optional): List of ints of month(s) (e.g., 1, 2,
                3...) to consider when calculating the wind condition
                frequencies. If none are specified, all months will be used.
                Defaults to None.
            limit_hour (list, optional): List of ints of hour(s) (e.g., 0, 1,
                ... 23) to consider when calculating the wind condition
                frequencies. If none are specified, all hours will be used.
                Defaults to None.
            st_date (str, optional): The start date to consider when creating
                the wind rose, formatted as 'MM-DD-YYYY'. If not specified data
                beginning in 2007 will be used. Defaults to None.
            en_date (str, optional): The end date to consider when creating
                the wind rose, formatted as 'MM-DD-YYYY'. If not specified data
                through 2013 will be used. Defaults to None.

        Returns:
            pandas.DataFrame: Wind rose DataFrame containing at least the
            following columns:

                - **wd** (*float*) - Wind direction bin center values (deg).
                - **ws** (*float*) - Wind speed bin center values (m/s).
                - **freq_val** (*float*) - The frequency of occurance of the
                  wind conditions in the other columns.
        """
        # Check inputs

        # Array of hub height data avaliable on Toolkit
        h_range = [10, 40, 60, 80, 100, 120, 140, 160, 200]

        if st_date is not None:
            if dateutil.parser.parse(st_date) > dateutil.parser.parse(
                "12-13-2013 23:00"
            ):
                print(
                    "Error, invalid date range. Valid range: 01-01-2007 - "
                    + "12/31/2013"
                )
                return None

        if en_date is not None:
            if dateutil.parser.parse(en_date) < dateutil.parser.parse(
                "01-01-2007 00:00"
            ):
                print(
                    "Error, invalid date range. Valid range: 01-01-2007 - "
                    + "12/31/2013"
                )
                return None

        if h_range[0] > ht:
            print(
                "Error, height is not in the range of avaliable "
                + "WindToolKit data. Minimum height = 10m"
            )
            return None

        if h_range[-1] < ht:
            print(
                "Error, height is not in the range of avaliable "
                + "WindToolKit data. Maxiumum height = 200m"
            )
            return None

        # Load wind speeds and directions from WimdToolkit

        # Case for turbine height (ht) matching discrete avaliable height
        # (h_range)
        if ht in h_range:

            d = self.load_wind_toolkit_hsds(
                lat,
                lon,
                ht,
                include_ti=include_ti,
                limit_month=limit_month,
                limit_hour=limit_hour,
                st_date=st_date,
                en_date=en_date,
            )

            ws_new = d["ws"]
            wd_new = d["wd"]
            if include_ti:
                ti_new = d["ti"]

        # Case for ht not matching discete height
        else:
            h_range_up = next(x[0] for x in enumerate(h_range) if x[1] > ht)
            h_range_low = h_range_up - 1
            h_up = h_range[h_range_up]
            h_low = h_range[h_range_low]

            # Load data for boundary cases of ht
            d_low = self.load_wind_toolkit_hsds(
                lat,
                lon,
                h_low,
                include_ti=include_ti,
                limit_month=limit_month,
                limit_hour=limit_hour,
                st_date=st_date,
                en_date=en_date,
            )

            d_up = self.load_wind_toolkit_hsds(
                lat,
                lon,
                h_up,
                include_ti=include_ti,
                limit_month=limit_month,
                limit_hour=limit_hour,
                st_date=st_date,
                en_date=en_date,
            )

            # Wind Speed interpolation
            ws_low = d_low["ws"]
            ws_high = d_up["ws"]

            ws_new = np.array(ws_low) * (
                1 - ((ht - h_low) / (h_up - h_low))
            ) + np.array(ws_high) * ((ht - h_low) / (h_up - h_low))

            # Wind Direction interpolation using Circular Mean method
            wd_low = d_low["wd"]
            wd_high = d_up["wd"]

            sin0 = np.sin(np.array(wd_low) * (np.pi / 180))
            cos0 = np.cos(np.array(wd_low) * (np.pi / 180))
            sin1 = np.sin(np.array(wd_high) * (np.pi / 180))
            cos1 = np.cos(np.array(wd_high) * (np.pi / 180))

            sin_wd = sin0 * (1 - ((ht - h_low) / (h_up - h_low))) + sin1 * (
                (ht - h_low) / (h_up - h_low)
            )
            cos_wd = cos0 * (1 - ((ht - h_low) / (h_up - h_low))) + cos1 * (
                (ht - h_low) / (h_up - h_low)
            )

            # Interpolated wind direction
            wd_new = 180 / np.pi * np.arctan2(sin_wd, cos_wd)

            # TI is independent of height
            if include_ti:
                ti_new = d_up["ti"]

        # Create a dataframe named df
        if include_ti:
            df = pd.DataFrame({"ws": ws_new, "wd": wd_new, "ti": ti_new})
        else:
            df = pd.DataFrame({"ws": ws_new, "wd": wd_new})

        # Start by simply round and wrapping the wind direction and wind speed
        # columns
        df["wd"] = geo.wrap_360(df.wd.round())
        df["ws"] = df.ws.round()

        # Now group up
        df["freq_val"] = 1.0
        df = df.groupby([c for c in df.columns if c != "freq_val"]).sum()
        df["freq_val"] = df.freq_val.astype(float) / df.freq_val.sum()
        df = df.reset_index()

        # Save the df at this point
        self.df = df

        # Resample onto the provided wind speed and wind direction binnings
        self.internal_resample_wind_speed(ws=ws)
        self.internal_resample_wind_direction(wd=wd)

        return self.df

    def load_wind_toolkit_hsds(
        self,
        lat,
        lon,
        ht=100,
        include_ti=False,
        limit_month=None,
        limit_hour=None,
        st_date=None,
        en_date=None,
    ):
        """
        This method returns a pandas DataFrame containing hourly wind speed,
        wind direction, and optionally estimated turbulence intensity data
        using wind data from the WIND Toolkit dataset
        (https://www.nrel.gov/grid/wind-toolkit.html) for the specified
        lat/long coordinate in the continental US. The wind data are obtained
        from the WIND Toolkit dataset using the HSDS service
        (see https://github.com/NREL/hsds-examples). The wind data returned is
        obtained from the nearest 2km x 2km grid point to the input coordinate
        and is limited to the years 2007-2013.

        TODO: make private method?

        Args:
            lat (float): Latitude in degrees.
            lon (float): Longitude in degrees
            ht (int, optional): The height above ground where wind
                information is obtained (m). Defaults to 100.
            include_ti (bool, optional): Determines whether turbulence
                intensity is included as an additional parameter. If True, TI
                is added as an additional wind rose variable, estimated based
                on the Obukhov length from WIND Toolkit. Defaults to False.
            limit_month (list, optional): List of ints of month(s) (e.g., 1, 2,
                3...) to consider when calculating the wind condition
                frequencies. If none are specified, all months will be used.
                Defaults to None.
            limit_hour (list, optional): List of ints of hour(s) (e.g., 0, 1,
                ... 23) to consider when calculating the wind condition
                frequencies. If none are specified, all hours will be used.
                Defaults to None.
            st_date (str, optional): The start date to consider, formatted as
                'MM-DD-YYYY'. If not specified data beginning in 2007 will be
                used. Defaults to None.
            en_date (str, optional): The end date to consider, formatted as
                'MM-DD-YYYY'. If not specified data through 2013 will be used.
                Defaults to None.

        Returns:
            pandas.DataFrame: DataFrame containing the following columns(abd
            optionally turbulence intensity) with hourly data from WIND Toolkit:

                - **wd** (*float*) - Raw wind direction data (deg).
                - **ws** (*float*) - Raw wind speed data (m/s).
        """
        import h5pyd

        # Open the wind data "file"
        # server endpoint, username, password is found via a config file
        f = h5pyd.File("/nrel/wtk-us.h5", "r")

        # assign wind direction, wind speed, optional ti, and time datasets for
        # the desired height
        wd_dset = f["winddirection_" + str(ht) + "m"]
        ws_dset = f["windspeed_" + str(ht) + "m"]
        if include_ti:
            obkv_dset = f["inversemoninobukhovlength_2m"]
        dt = f["datetime"]
        dt = pd.DataFrame({"datetime": dt[:]}, index=range(0, dt.shape[0]))
        dt["datetime"] = dt["datetime"].apply(dateutil.parser.parse)

        # find dataset indices from lat/long
        Location_idx = self.indices_for_coord(f, lat, lon)

        # check if in bounds
        if (
            (Location_idx[0] < 0)
            | (Location_idx[0] >= wd_dset.shape[1])
            | (Location_idx[1] < 0)
            | (Location_idx[1] >= wd_dset.shape[2])
        ):
            print(
                "Error, coordinates out of bounds. WIND Toolkit database "
                + "covers the continental United States."
            )
            return None

        # create dataframe with wind direction and wind speed
        df = pd.DataFrame()
        df["wd"] = wd_dset[:, Location_idx[0], Location_idx[1]]
        df["ws"] = ws_dset[:, Location_idx[0], Location_idx[1]]
        if include_ti:
            L = self.obkv_dset_to_L(obkv_dset, Location_idx)
            ti = self.ti_calculator_IU2(L)
            df["ti"] = ti
        df["datetime"] = dt["datetime"]

        # limit dates if start and end dates are provided
        if st_date is not None:
            df = df[df.datetime >= st_date]

        if en_date is not None:
            df = df[df.datetime < en_date]

        # limit to certain months if specified
        if limit_month is not None:
            df["month"] = df["datetime"].map(lambda x: x.month)
            df = df[df.month.isin(limit_month)]
        if limit_hour is not None:
            df["hour"] = df["datetime"].map(lambda x: x.hour)
            df = df[df.hour.isin(limit_hour)]
        if include_ti:
            df = df[["wd", "ws", "ti"]]
        else:
            df = df[["wd", "ws"]]

        return df

    def obkv_dset_to_L(self, obkv_dset, Location_idx):
        """
        This function returns an array containing hourly Obukhov lengths from
        the WIND Toolkit dataset for the specified Lat/Lon coordinate indices.

        Args:
            obkv_dset (np.ndarray): Dataset for Obukhov lengths from WIND
                Toolkit.
            Location_idx (tuple): A tuple containing the Lat/Lon coordinate
                indices of interest in the Obukhov length dataset.

        Returns:
            np.array: An array containing Obukhov lengths for each time index
            in the Wind Toolkit dataset (m).
        """
        linv = obkv_dset[:, Location_idx[0], Location_idx[1]]
        # avoid divide by zero
        linv[linv == 0.0] = 0.0003
        L = 1 / linv
        return L

    def ti_calculator_IU2(self, L):
        """
        This function estimates the turbulence intensity corresponding to each
        Obukhov length value in the input list using the relationship between
        Obukhov length bins and TI given in the I_U2SODAR column in Table 2 of
        :cite:`wr-wharton2010assessing`.

        Args:
            L (iterable): A list of Obukhov Length values (m).

        Returns:
            list: A list of turbulence intensity values expressed as fractions.
        """
        ti_set = []
        for i in L:
            # Strongly Stable
            if 0 < i < 100:
                TI = 0.04  # paper says < 8%, so using 4%
            # Stable
            elif 100 < i < 600:
                TI = 0.09
            # Neutral
            elif abs(i) > 600:
                TI = 0.115
            # Convective
            elif -600 < i < -50:
                TI = 0.165
            # Strongly Convective
            elif -50 < i < 0:
                # no upper bound given, so using the lowest
                # value from the paper for this stability bin
                TI = 0.2
            ti_set.append(TI)
        return ti_set

    def indices_for_coord(self, f, lat_index, lon_index):
        """
        This method finds the nearest x/y indices of the WIND Toolkit dataset
        for a given lat/lon coordinate in the continental US. Rather than
        fetching the entire coordinates database, which is 500+ MB, this uses
        the Proj4 library to find a nearby point and then converts to x/y
        indices.

        **Note**: This method is obtained directly from:
        https://github.com/NREL/hsds-examples/blob/master/notebooks/01_WTK_introduction.ipynb,
        where it is called "indicesForCoord."

        Args:
            f (h5pyd.File): A HDF5 "file" used to access the WIND Toolkit data.
            lat_index (float): Latitude coordinate for which dataset indices
                are to be found (degrees).
            lon_index (float): Longitude coordinate for which dataset indices
                are to be found (degrees).

        Returns:
            tuple: A tuple containing the Lat/Lon coordinate indices of
            interest in the WIND Toolkit dataset.
        """
        dset_coords = f["coordinates"]
        projstring = """+proj=lcc +lat_1=30 +lat_2=60
                        +lat_0=38.47240422490422 +lon_0=-96.0
                        +x_0=0 +y_0=0 +ellps=sphere
                        +units=m +no_defs """
        projectLcc = Proj(projstring)
        origin_ll = reversed(dset_coords[0][0])  # Grab origin directly from database
        origin = projectLcc(*origin_ll)

        coords = (lon_index, lat_index)
        coords = projectLcc(*coords)
        delta = np.subtract(coords, origin)
        ij = [int(round(x / 2000)) for x in delta]
        return tuple(reversed(ij))

    def plot_wind_speed_all(self, ax=None, label=None):
        """
        This method plots the wind speed frequency distribution of the WindRose
        object averaged across all wind directions. If no axis is provided, a
        new one is created.

        Args:
            ax (:py:class:`matplotlib.pyplot.axes`, optional): Figure axes on
                which data should be plotted. Defaults to None.
        """
        if ax is None:
            _, ax = plt.subplots()

        df_plot = self.df.groupby("ws").sum()
        ax.plot(self.ws, df_plot.freq_val, label=label)

    def plot_wind_speed_by_direction(self, dirs, ax=None):
        """
        This method plots the wind speed frequency distribution of the WindRose
        object for each specified wind direction bin center. The wind
        directions are resampled using the specified bin centers and the
        frequencies of occurance of the wind conditions are modified
        accordingly. If no axis is provided, a new one is created.

        Args:
            dirs (np.array): A list of wind direction bin centers for which
                wind speed distributions are plotted (deg).
            ax (:py:class:`matplotlib.pyplot.axes`, optional): Figure axes on
                which data should be plotted. Defaults to None.
        """
        # Get a downsampled frame
        df_plot = self.resample_wind_direction(self.df, wd=dirs)

        if ax is None:
            _, ax = plt.subplots()

        for wd in dirs:
            df_plot_sub = df_plot[df_plot.wd == wd]
            ax.plot(df_plot_sub.ws, df_plot_sub["freq_val"], label=wd)
        ax.legend()

    def plot_wind_rose(
        self,
        ax=None,
        color_map="viridis_r",
        ws_right_edges=np.array([5, 10, 15, 20, 25]),
        wd_bins=np.arange(0, 360, 15.0),
        legend_kwargs={},
    ):
        """
        This method creates a wind rose plot showing the frequency of occurance
        of the specified wind direction and wind speed bins. If no axis is
        provided, a new one is created.

        **Note**: Based on code provided by Patrick Murphy from the University
        of Colorado Boulder.

        Args:
            ax (:py:class:`matplotlib.pyplot.axes`, optional): The figure axes
                on which the wind rose is plotted. Defaults to None.
            color_map (str, optional): Colormap to use. Defaults to 'viridis_r'.
            ws_right_edges (np.array, optional): The upper bounds of the wind
                speed bins (m/s). The first bin begins at 0. Defaults to
                np.array([5, 10, 15, 20, 25]).
            wd_bins (np.array, optional): The wind direction bin centers used
                for plotting (deg). Defaults to np.arange(0, 360, 15.).
            legend_kwargs (dict, optional): Keyword arguments to be passed to
                ax.legend().

        Returns:
            :py:class:`matplotlib.pyplot.axes`: A figure axes object containing
            the plotted wind rose.
        """
        # Resample data onto bins
        df_plot = self.resample_wind_direction(self.df, wd=wd_bins)

        # Make labels for wind speed based on edges
        ws_step = ws_right_edges[1] - ws_right_edges[0]
        ws_labels = ["%d-%d m/s" % (w - ws_step, w) for w in ws_right_edges]

        # Grab the wd_step
        wd_step = wd_bins[1] - wd_bins[0]

        # Set up figure
        if ax is None:
            _, ax = plt.subplots(subplot_kw={"polar": True})

        # Get a color array
        color_array = cm.get_cmap(color_map, len(ws_right_edges))

        for wd in wd_bins:
            rects = []
            df_plot_sub = df_plot[df_plot.wd == wd]
            for ws_idx, ws in enumerate(ws_right_edges[::-1]):
                plot_val = df_plot_sub[
                    df_plot_sub.ws <= ws
                ].freq_val.sum()  # Get the sum of frequency up to this wind speed
                rects.append(
                    ax.bar(
                        np.radians(wd),
                        plot_val,
                        width=0.9 * np.radians(wd_step),
                        color=color_array(ws_idx),
                        edgecolor="k",
                    )
                )
            # break

        # Configure the plot
        ax.legend(reversed(rects), ws_labels, **legend_kwargs)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_theta_zero_location("N")
        ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

        return ax

    def plot_wind_rose_ti(
        self,
        ax=None,
        color_map="viridis_r",
        ti_right_edges=np.array([0.06, 0.1, 0.14, 0.18, 0.22]),
        wd_bins=np.arange(0, 360, 15.0),
    ):
        """
        This method creates a wind rose plot showing the frequency of occurance
        of the specified wind direction and turbulence intensity bins. This
        requires turbulence intensity to already be included as a parameter in
        the wind rose. If no axis is provided,a new one is created.

        **Note**: Based on code provided by Patrick Murphy from the University
        of Colorado Boulder.

        Args:
            ax (:py:class:`matplotlib.pyplot.axes`, optional): The figure axes
                on which the wind rose is plotted. Defaults to None.
            color_map (str, optional): Colormap to use. Defaults to 'viridis_r'.
            ti_right_edges (np.array, optional): The upper bounds of the
                turbulence intensity bins. The first bin begins at 0. Defaults
                to np.array([0.06, 0.1, 0.14, 0.18,0.22]).
            wd_bins (np.array, optional): The wind direction bin centers used
                for plotting (deg). Defaults to np.arange(0, 360, 15.).

        Returns:
            :py:class:`matplotlib.pyplot.axes`: A figure axes object containing
            the plotted wind rose.
        """

        # Resample data onto bins
        df_plot = self.resample_wind_direction(self.df, wd=wd_bins)

        # Make labels for TI based on edges
        ti_step = ti_right_edges[1] - ti_right_edges[0]
        ti_labels = ["%.2f-%.2f " % (w - ti_step, w) for w in ti_right_edges]

        # Grab the wd_step
        wd_step = wd_bins[1] - wd_bins[0]

        # Set up figure
        if ax is None:
            _, ax = plt.subplots(subplot_kw={"polar": True})

        # Get a color array
        color_array = cm.get_cmap(color_map, len(ti_right_edges))

        for wd in wd_bins:
            rects = []
            df_plot_sub = df_plot[df_plot.wd == wd]
            for ti_idx, ti in enumerate(ti_right_edges[::-1]):
                plot_val = df_plot_sub[
                    df_plot_sub.ti <= ti
                ].freq_val.sum()  # Get the sum of frequency up to this wind speed
                rects.append(
                    ax.bar(
                        np.radians(wd),
                        plot_val,
                        width=0.9 * np.radians(wd_step),
                        color=color_array(ti_idx),
                        edgecolor="k",
                    )
                )

        # Configure the plot
        ax.legend(reversed(rects), ti_labels, loc="lower right", title="TI")
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_theta_zero_location("N")
        ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

        return ax

    def plot_ti_ws(self, ax=None, ws_bins=np.arange(0, 26, 1.0)):
        """
        This method plots the wind speed frequency distribution of the WindRose
        object for each turbulence intensity bin. The wind speeds are resampled
        using the specified bin centers and the frequencies of occurance of the
        wind conditions are modified accordingly. This method assumes there are
        five TI bins. If no axis is provided, a new one is created.

        Args:
            ax (:py:class:`matplotlib.pyplot.axes`, optional): Figure axes on
                which data should be plotted. Defaults to None.
            ws_bins (np.array, optional): A list of wind speed bin centers on
                which the wind speeds are resampled before plotting (m/s).
                Defaults to np.arange(0, 26, 1.).

        Returns:
            :py:class:`matplotlib.pyplot.axes`: A figure axes object containing
            the plotted wind speed distributions.
        """

        # Resample data onto bins
        df_plot = self.resample_wind_speed(self.df, ws=ws_bins)

        df_plot = df_plot.groupby(["ws", "ti"]).sum()
        df_plot = df_plot.reset_index()

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 7))

        tis = df_plot["ti"].drop_duplicates()
        margin_bottom = np.zeros(len(df_plot["ws"].drop_duplicates()))
        colors = ["#1e5631", "#a4de02", "#76ba1b", "#4c9a2a", "#acdf87"]

        for num, ti in enumerate(tis):
            values = list(df_plot[df_plot["ti"] == ti].loc[:, "freq_val"])

            df_plot[df_plot["ti"] == ti].plot.bar(
                x="ws",
                y="freq_val",
                ax=ax,
                bottom=margin_bottom,
                color=colors[num],
                label=ti,
            )

            margin_bottom += values

        plt.title("Turbulence Intensity Frequencies as Function of Wind Speed")
        plt.xlabel("Wind Speed (m/s)")
        plt.ylabel("Frequency")

        return ax

    def export_for_floris_opt(self):
        """
        This method returns a list of tuples of at least wind speed, wind
        direction, and frequency of occurance, which can be used to help loop
        through different wind conditions for Floris power calculations.

        Returns:
            list: A list of tuples containing all combinations of wind
            parameters and frequencies of occurance in the WindRose object's
            wind rose DataFrame values.
        """
        # Return a list of tuples, where each tuple is (ws,wd,freq)
        return [tuple(x) for x in self.df.values]
