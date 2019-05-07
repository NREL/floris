# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

## ISSUES TO TACKLE
## 1: BINNING VALUES OUTSIDE OF WD AND WS (just ignore?)
## 2: Include smoothing?
## 3: Plotting
## 4: FLORIS return

import os
import dateutil
import h5pyd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import pickle
from pyproj import Proj
import floris.utilities as geo


class WindRose():
    """
    WindRose object class used to parse data and generate figures.
    """

    def __init__(self, ):
        """
        Init Function of WindRose Object.
        No explicit arguments required.
        """

        # Initialize some varibles to zero
        self.num_wd = 0
        self.num_ws = 0
        self.wd_step = 1.
        self.ws_step = 5.
        self.wd = np.array([])
        self.ws = np.array([])
        self.df = pd.DataFrame()

    def save(self, filename):
        """
        Method for pickling WIndRose objects.

        Args:
            filename (str): Write-to path for WindRose pickle.
        """
        pickle.dump([
            self.num_wd, self.num_ws, self.wd_step, self.ws_step, self.wd,
            self.ws, self.df
        ], open(filename, "wb"))

    def load(self, filename):
        """
        Load previously pickled WindRose object.

        Args:
            filename (str): Read-from path for pickled WindRose Object
        """
        self.num_wd, self.num_ws, self.wd_step, self.ws_step, self.wd, self.ws, self.df = pickle.load(
            open(filename, "rb"))

    def resample_wind_speed(self, df, ws=np.arange(0, 26, 1.)):
        """
        Modify the default bins for sorting wind speed.

        Args:
            df (pd.DataFrame): Wind speed data
            ws (np.array, optional): Vector of wind speed bins for
                WindRose. Defaults to np.arange(0, 26, 1.).

        Returns:
            df (pd.DataFrame): Resampled wind speed for WindRose
        """
        # Make a copy of incoming dataframe
        df = df.copy(deep=True)

        # Get the wind step
        ws_step = ws[1] - ws[0]

        # Ws
        ws_edges = (ws - ws_step / 2.0)
        ws_edges = np.append(ws_edges, np.array(ws[-1] + ws_step / 2.0))

        # Cut wind speed onto bins
        df['ws'] = pd.cut(df.ws, ws_edges, labels=ws)

        # Regroup
        df = df.groupby(['ws', 'wd']).sum()

        # Fill nans
        df = df.fillna(0)

        # Reset the index
        df = df.reset_index()

        # Set to float
        df['ws'] = df.ws.astype(float)
        df['wd'] = df.wd.astype(float)

        return df

    def internal_resample_wind_speed(self, ws=np.arange(0, 26, 1.)):
        """
        Internal method for resampling wind speed into desired bins.
        Modifies data within WindRose object without explicit return.

        Args:
            ws (np.array, optional): Vector of wind speed bins for
                WindRose. Defaults to np.arange(0, 26, 1.).
        """
        # Update ws and wd binning
        self.ws = ws
        self.num_ws = len(ws)
        self.ws_step = ws[1] - ws[0]

        # Update internal data frame
        self.df = self.resample_wind_speed(self.df, ws)

    def resample_wind_direction(self, df, wd=np.arange(0, 360, 5.)):
        """
        Modify the default bins for sorting wind direction.

        Args:
            df (pd.DataFrame): Wind direction data
                wd (np.array, optional): Vector of wind direction bins
                for WindRose. Defaults to np.arange(0, 360, 5.).

        Returns:
            df (pd.DataFrame): Resampled wind direction for WindRose
        """
        # Make a copy of incoming dataframe
        df = df.copy(deep=True)

        # Get the wind step
        wd_step = wd[1] - wd[0]

        # Get bin edges
        wd_edges = (wd - wd_step / 2.0)
        wd_edges = np.append(wd_edges, np.array(wd[-1] + wd_step / 2.0))

        # Get the overhangs
        negative_overhang = wd_edges[0]
        positive_overhang = wd_edges[-1] - 360.

        # Need potentially to wrap high angle direction to negative for correct binning
        df['wd'] = geo.wrap_360(df.wd)
        if negative_overhang < 0:
            print('Correcting negative Overhang:%.1f' % negative_overhang)
            df['wd'] = np.where(df.wd.values >= 360. + negative_overhang,
                                df.wd.values - 360., df.wd.values)

        # Check on other side
        if positive_overhang > 0:
            print('Correcting positive Overhang:%.1f' % positive_overhang)
            df['wd'] = np.where(df.wd.values <= positive_overhang,
                                df.wd.values + 360., df.wd.values)

        # Cut into bins
        df['wd'] = pd.cut(df.wd, wd_edges, labels=wd)

        # Regroup
        df = df.groupby(['ws', 'wd']).sum()

        # Fill nans
        df = df.fillna(0)

        # Reset the index
        df = df.reset_index()

        # Set to float Re-wrap
        df['wd'] = df.wd.astype(float)
        df['ws'] = df.ws.astype(float)
        df['wd'] = geo.wrap_360(df.wd)

        return df

    def internal_resample_wind_direction(self, wd=np.arange(0, 360, 5.)):
        """
        Internal method for resampling wind direction into desired bins.
        Modifies data within WindRose object without explicit return.

        Args:
            wd (np.array, optional): Vector of wind direction bins for
                WindRose. Defaults to np.arange(0, 360, 5.).
        """
        # Update ws and wd binning
        self.wd = wd
        self.num_wd = len(wd)
        self.wd_step = wd[1] - wd[0]

        # Update internal data frame
        self.df = self.resample_wind_direction(self.df, wd)

    def resample_average_ws_by_wd(self, df):
        """
        Re-established counts of wind speed observations in wind
        direction bins.

        Args:
            df (pd.DataFrame): Wind speed and direction data

        Returns:
            df (pd.DataFrame): Resampled wind speed and direction data.
        """
        # Make a copy of incoming dataframe
        df = df.copy(deep=True)

        ws_avg = []

        for val in df.wd.unique():
            ws_avg.append(
                np.array(df.loc[df['wd'] == val]['ws'] *
                         df.loc[df['wd'] == val]['freq_val']).sum() /
                df.loc[df['wd'] == val]['freq_val'].sum())

        # Regroup
        df = df.groupby('wd').sum()

        df['ws'] = ws_avg

        # Reset the index
        df = df.reset_index()

        # Set to float
        df['ws'] = df.ws.astype(float)
        df['wd'] = df.wd.astype(float)

        return df

    def internal_resample_average_ws_by_wd(self, wd=np.arange(0, 360, 5.)):
        """
        Internal method for re-established counts of wind speed
        observations in wind direction bins.
        Modifies data within WindRose Object without explicit return.

        Args:
            wd (np.arange, optional): Wind direction bins limists.
                Defaults to np.arange(0, 360, 5.).
        """
        # Update ws and wd binning
        self.wd = wd
        self.num_wd = len(wd)
        self.wd_step = wd[1] - wd[0]
        # self #TODO This reference to 'self' was here alone. Is this intentional as a print? is it incomplete?

        # Update internal data frame
        self.df = self.resample_average_ws_by_wd(self.df)

    def weibull(self, x, k=2.5, lam=8.0):
        """
        Weibull distribution function object for least-squares fitting.

        Args:
            x (np.array): Input data (typically binned wind speed
                observations.)
            k (float, optional): Weibull share parameter.
                Defaults to 2.5.
            lam (float, optional): Weibull scale parameter.
                Defaults to 8.0.

        Returns:
            np.array: Weibull distribution
        """
        return (k / lam) * (x / lam)**(k - 1) * np.exp(-(x / lam)**k)

    def make_wind_rose_from_weibull(self,
                                    wd=np.arange(0, 360, 5.),
                                    ws=np.arange(0, 26, 1.)):
        """
        Populate binned observations of wind speed and direction from
        fitted Weibull distribution.

        Args:
            wd (np.array, optional): Wind direciton bins.
                Defaults to np.arange(0, 360, 5.).
            ws (np.array, optional): Wind speed bins.
                Defaults to np.arange(0, 26, 1.).

        Returns:
            df (pd.DataFrame): updated wind speed and direction bins.
            #TODO should these be returned or updated internally, both?
        """
        # Use an assumed wind-direction for dir frequency
        wind_dir = [
            0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5,
            270, 292.5, 315, 337.5
        ]
        freq_dir = [
            0.064, 0.04, 0.038, 0.036, 0.045, 0.05, 0.07, 0.08, 0.11, 0.08,
            0.05, 0.036, 0.048, 0.058, 0.095, 0.10
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

        # Start by simply round and wrapping the wind direction and wind speed columns
        df['wd'] = wd_tot
        df['ws'] = ws_tot

        # Now group up
        df['freq_val'] = freq_tot

        # Save the df at this point
        self.df = df
        #TODO is there a reason self.df is updated AND returned?
        return self.df

    def parse_wind_toolkit_folder(self,
                                  folder_name,
                                  wd=np.arange(0, 360, 5.),
                                  ws=np.arange(0, 26, 1.),
                                  limit_month=None):
        """
        Load and

        Args:
            folder_name (str): path to wind toolkit input data.
            wd (np.array, optional): Wind direction bin limits.
                Defaults to np.arange(0, 360, 5.).
            ws (np.array, optional): Wind speed bin limits.
                Defaults to np.arange(0, 26, 1.).
            limit_month (str, optional): name of month(s) to consider.
                Defaults to None.

        Returns:
            df (pd.DataFrame): DataFrame updated with wind toolkit info.
        """
        # Load the wind toolkit data into a dataframe
        df = self.load_wind_toolkit_folder(folder_name,
                                           limit_month=limit_month)

        # Start by simply round and wrapping the wind direction and wind speed columns
        df['wd'] = geo.wrap_360(df.wd.round())
        df['ws'] = geo.wrap_360(df.ws.round())

        # Now group up
        df['freq_val'] = 1.
        df = df.groupby(['ws', 'wd']).sum()
        df['freq_val'] = df.freq_val.astype(float) / df.freq_val.sum()
        df = df.reset_index()

        # Save the df at this point
        self.df = df

        # Resample onto the provided wind speed and wind direction binnings
        self.internal_resample_wind_speed(ws=ws)
        self.internal_resample_wind_direction(wd=wd)

        return self.df

    def load_wind_toolkit_folder(self, folder_name, limit_month=None):
        """
        Given a wind_toolkit folder of files, produce a list of
        files to input

        Args:
            folder_name (str): path to folder containing wind toolikit
                data
            limit_month (str, optional): limit loaded data to specified
                month. Defaults to None.

        Returns:
            df (pd.DataFrame): DataFrame containing wind data from each
                of the files in the folder.
        """

        file_list = os.listdir(folder_name)
        file_list = [
            os.path.join(folder_name, f) for f in file_list if '.csv' in f
        ]

        df = pd.DataFrame()
        for f_idx, f in enumerate(file_list):
            print('%d of %d: %s' % (f_idx, len(file_list), f))
            df_temp = self.load_wind_toolkit_file(f, limit_month=limit_month)
            df = df.append(df_temp)

        return df

    def load_wind_toolkit_file(self, filename, limit_month=None):
        """
        Load a particular wind toolkit data file.

        Args:
            filename (str): path to data file.
            limit_month (str, optional): limit loaded data to specified
                month. Defaults to None.

        Returns:
            df (pd.DataFrame): DataFrame containing wind data from the
                specified file.
        """
        df = pd.read_csv(filename, header=3, sep=',')

        # If asked to limit to particular months
        if not limit_month is None:
            df = df[df.Month.isin(limit_month)]

        # Save just what I want
        speed_column = [c for c in df.columns if 'speed' in c][0]
        direction_column = [c for c in df.columns if 'direction' in c][0]
        df = df.rename(index=str,
                       columns={
                           speed_column: "ws",
                           direction_column: "wd"
                       })[['wd', 'ws']]

        return df

    def import_from_wind_toolkit_hsds(self,
                                      lat,
                                      lon,
                                      ht=100,
                                      wd=np.arange(0, 360, 5.),
                                      ws=np.arange(0, 26, 1.),
                                      limit_month=None,
                                      st_date=None,
                                      en_date=None):
        """
        Given a lat/long coordinate in the continental US return a
        dataframe containing the normalized frequency of each pair of
        wind speed and wind direction values specified. The wind data
        is obtained from the WIND Toolkit dataset (https://www.nrel.gov/
        grid/wind-toolkit.html) using the HSDS service (see https://
        github.com/NREL/hsds-examples). The wind data returned is
        obtained from the nearest 2km x 2km grid point to the input
        coordinate and is limited to the years 2007-2013.

        Requires h5pyd package, which can be installed using:
            pip install --user git+http://github.com/HDFGroup/h5pyd.git

        Then, make a configuration file at ~/.hscfg containing:

            hs_endpoint = https://developer.nrel.gov/api/hsds/
            hs_username = None
            hs_password = None
            hs_api_key = 3K3JQbjZmWctY0xmIfSYvYgtIcM3CN0cb1Y2w9bf

        The example API key above is for demonstation and is
        rate-limited per IP. To get your own API key, visit https://
        developer.nrel.gov/signup/

        More information can be found at: https://github.com/NREL/
        hsds-examples

        Args:
            lat (float): coordinates in degrees
            lon (float): coordinates in degrees
            ht (int, optional): height above ground where wind
                information is obtained (m). Defaults to 100.
            wd (np.array, optional): Wind direction bin limits.
                Defaults to np.arange(0, 360, 5.).
            ws (np.array, optional): Wind speed bin limits.
                Defaults to np.arange(0, 26, 1.).
            limit_month (str, optional): limit loaded data to specified
                month. Defaults to None.
            st_date (str, optional): 'MM-DD-YYYY'.
                Defaults to None.
            en_date (str, optional): 'MM-DD-YYYY'.
                Defaults to None.

        Returns:
            df (pd.DataFrame): DataFrame with wind speed and direction
                data.
        """

        # check inputs
        if st_date is not None:
            if dateutil.parser.parse(st_date) > dateutil.parser.parse(
                    '12-13-2013 23:00'):
                print(
                    'Error, invalid date range. Valid range: 01-01-2007 - 12/31/2013'
                )
                return None
        if en_date is not None:
            if dateutil.parser.parse(en_date) < dateutil.parser.parse(
                    '01-01-2007 00:00'):
                print(
                    'Error, invalid date range. Valid range: 01-01-2007 - 12/31/2013'
                )
                return None

        if ht not in [10, 40, 60, 80, 100, 120, 140, 160, 200]:
            print(
                'Error, invalid height. Valid heights: 10, 40, 60, 80, 100, 120, 140, 160, 200'
            )
            return None

        # load wind speeds and directions from WIND Toolkit
        df = self.load_wind_toolkit_hsds(lat,
                                         lon,
                                         ht=ht,
                                         limit_month=limit_month,
                                         st_date=st_date,
                                         en_date=en_date)

        # check for errors in loading wind toolkit data
        if df is None:
            return None

        # Start by simply round and wrapping the wind direction and wind speed columns
        df['wd'] = geo.wrap_360(df.wd.round())
        df['ws'] = geo.wrap_360(df.ws.round())

        # Now group up
        df['freq_val'] = 1.
        df = df.groupby(['ws', 'wd']).sum()
        df['freq_val'] = df.freq_val.astype(float) / df.freq_val.sum()
        df = df.reset_index()

        # Save the df at this point
        self.df = df

        # Resample onto the provided wind speed and wind direction binnings
        self.internal_resample_wind_speed(ws=ws)
        self.internal_resample_wind_direction(wd=wd)

        return df

    def load_wind_toolkit_hsds(self,
                               lat,
                               lon,
                               ht=100,
                               limit_month=None,
                               st_date=None,
                               en_date=None):
        """
        Given a lat/long coordinate in the continental US return a
        dataframe containing wind speeds and wind directions at the
        specified height for each hour in the date range given. The
        wind data is obtained from the WIND Toolkit dataset (https://
        www.nrel.gov/grid/wind-toolkit.html) using the HSDS service
        (see https://github.com/NREL/hsds-examples). The wind data
        returned is obtained from the nearest 2km x 2km grid point to
        the input coordinate.

        Args:
            lat (float): coordinates in degrees.
            lon (float): coordinates in degrees.
            ht (int, optional): height above ground where wind
                information is obtained (m).
                Defaults to 100.
            limit_month (str, optional): limit loaded data to specified
                month. Defaults to None.
            st_date (str, optional): 'MM-DD-YYYY'.
                Defaults to None.
            en_date (str, optional): 'MM-DD-YYYY'.
                Defaults to None.

        Returns:
            df (pd.DataFrame): dataframe with wind speed and wind
                directiond columns containing hourly data.
        """

        # Open the wind data "file"
        # server endpoint, username, password is found via a config file
        f = h5pyd.File("/nrel/wtk-us.h5", 'r')

        # assign wind direction, wind speed, and time datasets for the desired height
        wd_dset = f['winddirection_' + str(ht) + 'm']
        ws_dset = f['windspeed_' + str(ht) + 'm']
        dt = f['datetime']
        dt = pd.DataFrame({'datetime': dt[:]}, index=range(0, dt.shape[0]))
        dt['datetime'] = dt['datetime'].apply(dateutil.parser.parse)

        # find dataset indices from lat/long
        Location_idx = self.indices_for_coord(f, lat, lon)

        # check if in bounds
        if (Location_idx[0] < 0) | (Location_idx[0] >= wd_dset.shape[1]) | (
                Location_idx[1] < 0) | (Location_idx[1] >= wd_dset.shape[2]):
            print(
                'Error, coordinates out of bounds. WIND Toolkit database covers the continental United States.'
            )
            return None

        # create dataframe with wind direction and wind speed
        df = pd.DataFrame()
        df['wd'] = wd_dset[:, Location_idx[0], Location_idx[1]]
        df['ws'] = ws_dset[:, Location_idx[0], Location_idx[1]]
        df['datetime'] = dt['datetime']

        # limit dates if start and end dates are provided
        if (st_date is not None):
            df = df[df.datetime >= st_date]

        if (en_date is not None):
            df = df[df.datetime < en_date]

        # limit to certain months if specified
        if not limit_month is None:
            df['month'] = df['datetime'].map(lambda x: x.month)
            df = df[df.month.isin(limit_month)]
        df = df[['wd', 'ws']]

        return df

    def indices_for_coord(self, f, lat_index, lon_index):
        #TODO This function is tough for me to follow. What is f?
        """
        Function obtained from: https://github.com/NREL/hsds-examples/
        blob/master/notebooks/01_introduction.ipynb "indicesForCoord"
        This function finds the nearest x/y indices for a given lat/lon.
        Rather than fetching the entire coordinates database, which is
        500+ MB, this uses the Proj4 library to find a nearby point and
        then converts to x/y indices.

        Args:
            f (dict): #TODO [description]
            lat_index (int): index to desired latitude coordinate.
            lon_index (int): index to desired longitude coordinate.

        Returns:
            tuple: #TODO [description]
        """

        dset_coords = f['coordinates']
        projstring = """+proj=lcc +lat_1=30 +lat_2=60
                        +lat_0=38.47240422490422 +lon_0=-96.0
                        +x_0=0 +y_0=0 +ellps=sphere
                        +units=m +no_defs """
        projectLcc = Proj(projstring)
        origin_ll = reversed(
            dset_coords[0][0])  # Grab origin directly from database
        origin = projectLcc(*origin_ll)

        coords = (lon_index, lat_index)
        coords = projectLcc(*coords)
        delta = np.subtract(coords, origin)
        ij = [int(round(x / 2000)) for x in delta]
        return tuple(reversed(ij))

    def plot_wind_speed_all(self, ax=None):
        """
        Add binned wind speed observations to a plot. If no axis is
        provided, make a new one.

        Args:
            ax (:py:class:`matplotlib.pyplot.axes`, optional): Figure
                axes to which data should be plotted. Defaults to None.
        """
        if ax is None:
            _, ax = plt.subplots()

        df_plot = self.df.groupby('ws').sum()
        ax.plot(self.ws, df_plot.freq_val)

    def plot_wind_speed_by_direction(self, dirs, ax=None):
        """
        Add  wind speed observations binned by wind direction to a
        plot. If no axis is provided, make a new one.

        Args:
            dirs (np.array): vector of wind direction bins.
            ax (:py:class:`matplotlib.pyplot.axes`, optional): Figure
                axes to which data should be plotted. Defaults to None.
        """
        # Get a downsampled frame
        df_plot = self.resample_wind_direction(self.df, wd=dirs)

        if ax is None:
            _, ax = plt.subplots()

        for wd in dirs:
            df_plot_sub = df_plot[df_plot.wd == wd]
            ax.plot(df_plot_sub.ws, df_plot_sub['freq_val'], label=wd)
        ax.legend()

    def plot_wind_rose(self,
                       ax=None,
                       color_map='viridis_r',
                       ws_right_edges=np.array([5, 10, 15, 20, 25]),
                       wd_bins=np.arange(0, 360, 15.)):
        """
        Generate wind rose plot. If no axis is provided, make a new one.

        Args:
            ax (:py:class:`matplotlib.pyplot.axes`, optional): Figure axes to which data should
                be plotted. Defaults to None.
            color_map (str, optional): name of colormap.
                Defaults to 'viridis_r'.
            ws_right_edges (np.array, optional): upper bounds of wind
                speed bins. Defaults to np.array([5, 10, 15, 20, 25]).
            wd_bins (np.array, optional): wind direction bin limits.
                Defaults to np.arange(0, 360, 15.).

        Returns:
            ax (:py:class:`matplotlib.pyplot.axes`): Figure axes
                containing wind rose plot.
        """
        # Based on code provided by Patrick Murphy

        # Resample data onto bins
        # df_plot = self.resample_wind_speed(self.df,ws=ws_bins)
        df_plot = self.resample_wind_direction(self.df, wd=wd_bins)

        # Make labels for wind speed based on edges
        ws_step = ws_right_edges[1] - ws_right_edges[0]
        # ws = ws_edges
        # ws_edges = (ws - ws_step / 2.0)
        # ws_edges = np.append(ws_edges,np.array(ws[-1] + ws_step / 2.0))
        # ws_edges = np.append([0], ws_right_edges)
        ws_labels = ['%d-%d m/s' % (w - ws_step, w) for w in ws_right_edges]

        # Grab the wd_step
        wd_step = wd_bins[1] - wd_bins[0]

        # Set up figure
        if ax is None:
            _, ax = plt.subplots(subplot_kw=dict(polar=True))

        # Get a color array
        color_array = cm.get_cmap(color_map, len(ws_right_edges))

        for wd_idx, wd in enumerate(wd_bins):
            rects = list()
            df_plot_sub = df_plot[df_plot.wd == wd]
            for ws_idx, ws in enumerate(ws_right_edges[::-1]):
                plot_val = df_plot_sub[df_plot_sub.ws <= ws].freq_val.sum(
                )  # Get the sum of frequency up to this wind speed
                rects.append(
                    ax.bar(np.radians(wd),
                           plot_val,
                           width=0.9 * np.radians(wd_step),
                           color=color_array(ws_idx),
                           edgecolor='k'))
            # break

        # Configure the plot
        ax.legend(reversed(rects), ws_labels)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_theta_zero_location("N")
        ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

        return ax

    def export_for_floris_opt(self):
        """
        Shortcut function to generate output for FLORIS.

        Returns:
            (list): tuples containing DataFrame values.
        """
        # Return a list of tuples, where each tuple is (ws,wd,freq)
        return [tuple(x) for x in self.df.values]