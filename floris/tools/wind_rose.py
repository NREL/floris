"""
Copyright 2018 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

## ISSUES TO TACKLE
## 1: BINNING VALUES OUTSIDE OF WD AND WS (just ignore?)
## 2: Include smoothing?
## 3: Plotting
## 4: FLORIS return


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import wind_tools.geometry as geo
# from windrose import WindroseAxes
import matplotlib.cm as cm
import h5pyd
import dateutil
from pyproj import Proj
import pickle

class WindRose():

    def __init__(self,):
        """Init Function, maybe nothing to do, not sure


        input: 
            
        output:
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
        pickle.dump( [self.num_wd,self.num_ws,self.wd_step,self.ws_step,self.wd,self.ws,self.df ], open( filename, "wb" ) )

    def load(self, filename):
        self.num_wd,self.num_ws,self.wd_step,self.ws_step,self.wd,self.ws,self.df  = pickle.load( open( filename, "rb" ) )

    def resample_wind_speed(self,df,ws=np.arange(0,26,1.) ):
        
        # Make a copy of incoming dataframe
        df = df.copy(deep=True)

        # Get the wind step
        ws_step = ws[1] - ws[0]

        # Ws
        ws_edges = (ws - ws_step / 2.0)
        ws_edges = np.append(ws_edges,np.array(ws[-1] + ws_step / 2.0))

        # Cut wind speed onto bins
        df['ws'] = pd.cut(df.ws,ws_edges,labels=ws)

        # Regroup
        df = df.groupby(['ws','wd']).sum()

        # Fill nans
        df = df.fillna(0)

        # Reset the index
        df = df.reset_index()

        # Set to float
        df['ws'] = df.ws.astype(float)
        df['wd'] = df.wd.astype(float)

        return df


    def internal_resample_wind_speed(self,ws=np.arange(0,26,1.) ):
        
        # Update ws and wd binning
        self.ws = ws
        self.num_ws = len(ws)
        self.ws_step = ws[1] - ws[0]

        # Update internal data frame
        self.df = self.resample_wind_speed(self.df,ws)


    def resample_wind_direction(self,df,wd = np.arange(0,360,5.) ):
        
        # Make a copy of incoming dataframe
        df = df.copy(deep=True)

        # Get the wind step
        wd_step = wd[1] - wd[0]

        # Get bin edges
        wd_edges = (wd - wd_step / 2.0)
        wd_edges = np.append(wd_edges,np.array(wd[-1] + wd_step / 2.0))

        # Get the overhangs
        negative_overhang = wd_edges[0]
        positive_overhang = wd_edges[-1] - 360.

        # Need potentially to wrap high angle direction to negative for correct binning
        df['wd'] = geo.wrap_360(df.wd)
        if negative_overhang < 0:
            print('Correcting negative Overhang:%.1f' % negative_overhang)
            df['wd'] = np.where(df.wd.values>=360. + negative_overhang,df.wd.values-360.,df.wd.values)

        # Check on other side
        if positive_overhang > 0:
            print('Correcting positive Overhang:%.1f' % positive_overhang)
            df['wd'] = np.where(df.wd.values<= positive_overhang,df.wd.values+360.,df.wd.values)

        # Cut into bins
        df['wd'] = pd.cut(df.wd,wd_edges,labels=wd)

        # Regroup
        df = df.groupby(['ws','wd']).sum()

        # Fill nans
        df = df.fillna(0)

        # Reset the index
        df = df.reset_index()

        # Set to float Re-wrap
        df['wd'] = df.wd.astype(float)
        df['ws'] = df.ws.astype(float)
        df['wd'] = geo.wrap_360(df.wd)

        return df


    def internal_resample_wind_direction(self,wd = np.arange(0,360,5.) ):
        
        # Update ws and wd binning
        self.wd = wd
        self.num_wd = len(wd)
        self.wd_step = wd[1] - wd[0]

        # Update internal data frame
        self.df = self.resample_wind_direction(self.df,wd)


    def resample_average_ws_by_wd(self, df):

        # Make a copy of incoming dataframe
        df = df.copy(deep=True)

        ws_avg = []

        for val in df.wd.unique():
            ws_avg.append(np.array(df.loc[df['wd'] == val]['ws']*df.loc[df['wd'] == val]['freq_val']).sum()/df.loc[df['wd'] == val]['freq_val'].sum())
            
        # Regroup
        df = df.groupby('wd').sum()

        df['ws'] = ws_avg

        # Reset the index
        df = df.reset_index()

        # Set to float
        df['ws'] = df.ws.astype(float)
        df['wd'] = df.wd.astype(float)

        return df


    def internal_resample_average_ws_by_wd(self,wd = np.arange(0,360,5.) ):
        
        # Update ws and wd binning
        self.wd = wd
        self.num_wd = len(wd)
        self.wd_step = wd[1] - wd[0]
        self

        # Update internal data frame
        self.df = self.resample_average_ws_by_wd(self.df)


    def weibull(self,x,k=2.5,lam=8.0):  
        return (k/lam) * (x/lam)**(k-1) * np.exp(-(x/lam)**k)

    def make_wind_rose_from_weibull(self, wd = np.arange(0,360,5.), ws=np.arange(0,26,1.)):

        # Use an assumed wind-direction for dir frequency
        wind_dir = [0,   22.5,45,   67.5, 90,   112.5,135, 157.5,180, 202.5,225, 247.5,270,  292.5,315,  337.5]
        freq_dir = [0.064,0.04,0.038,0.036,0.045,0.05, 0.07,0.08, 0.11,0.08, 0.05,0.036,0.048,0.058,0.095,0.10]

        freq_wd = np.interp(wd, wind_dir, freq_dir)
        freq_ws = self.weibull(ws)

        freq_tot = np.zeros(len(wd)*len(ws))
        wd_tot = np.zeros(len(wd)*len(ws))
        ws_tot = np.zeros(len(wd)*len(ws))

        count = 0
        for i in range(len(wd)):
            for j in range(len(ws)):
                wd_tot[count] = wd[i]
                ws_tot[count] = ws[j]
                    
                freq_tot[count] = freq_wd[i]*freq_ws[j]
                count = count + 1

        # renormalize
        freq_tot = freq_tot/np.sum(freq_tot)

        # Load the wind toolkit data into a dataframe
        df = pd.DataFrame()

        # Start by simply round and wrapping the wind direction and wind speed columns
        df['wd'] = wd_tot
        df['ws'] = ws_tot

        # Now group up
        df['freq_val'] = freq_tot

        # Save the df at this point
        self.df = df

        return self.df

    def parse_wind_toolkit_folder(self, folder_name, wd = np.arange(0,360,5.), ws=np.arange(0,26,1.), limit_month=None):

        # Load the wind toolkit data into a dataframe
        df = self.load_wind_toolkit_folder(folder_name, limit_month=limit_month)

        # Start by simply round and wrapping the wind direction and wind speed columns
        df['wd'] = geo.wrap_360(df.wd.round())
        df['ws'] = geo.wrap_360(df.ws.round())

        # Now group up
        df['freq_val'] = 1.
        df = df.groupby(['ws','wd']).sum()
        df['freq_val'] = df.freq_val.astype(float) / df.freq_val.sum()
        df = df.reset_index()

        # Save the df at this point
        self.df = df

        # Resample onto the provided wind speed and wind direction binnings
        self.internal_resample_wind_speed(ws=ws)
        self.internal_resample_wind_direction(wd=wd)


        return self.df




    def load_wind_toolkit_folder(self, folder_name, limit_month=None):
        """ Given a wind_toolkit folder of files, produce a list of files to input
        input: folder name
        output: list of files, with folder name included

        """
        file_list = os.listdir(folder_name)
        file_list = [os.path.join(folder_name,f) for f in file_list if '.csv' in f]
        
        df = pd.DataFrame()
        for f_idx, f in enumerate(file_list):
            print('%d of %d: %s' % (f_idx, len(file_list),f))
            df_temp = self.load_wind_toolkit_file(f, limit_month=limit_month)
            df = df.append(df_temp)

        return df

    def load_wind_toolkit_file(self, filename, limit_month=None):
        df = pd.read_csv(filename,header=3,sep=',')

        # If asked to limit to particular months
        if not limit_month is None:
            df = df[df.Month.isin(limit_month)]

        # Save just what I want
        speed_column = [c for c in df.columns if 'speed' in c][0]
        direction_column = [c for c in df.columns if 'direction' in c][0]
        df = df.rename(index=str, columns={speed_column: "ws", direction_column: "wd"})[['wd','ws']]

        return df

    def import_from_wind_toolkit_hsds(self, lat, lon, ht = 100, wd = np.arange(0,360,5.), ws=np.arange(0,26,1.), limit_month=None, st_date=None, en_date=None):
        """ 
        Given a lat/long coordinate in the continental US return a dataframe containing the normalized 
        frequency of each pair of wind speed and wind direction values specified. The wind data is obtained 
        from the WIND Toolkit dataset (https://www.nrel.gov/grid/wind-toolkit.html) using the HSDS service 
        (see https://github.com/NREL/hsds-examples). The wind data returned is obtained from the nearest 
        2km x 2km grid point to the input coordinate and is limited to the years 2007-2013.

        Requires h5pyd package, which can be installed using:
            pip install --user git+http://github.com/HDFGroup/h5pyd.git

        Then, make a configuration file at ~/.hscfg containing:

            hs_endpoint = https://developer.nrel.gov/api/hsds/
            hs_username = None
            hs_password = None
            hs_api_key = 3K3JQbjZmWctY0xmIfSYvYgtIcM3CN0cb1Y2w9bf

        The example API key above is for demonstation and is rate-limited per IP. To get your own API key, visit https://developer.nrel.gov/signup/

        More information can be found at: https://github.com/NREL/hsds-examples

        input: lat, lon: coordinates in degrees, ht: height above ground where wind information is 
        obtained (m), wd, ws: arrays containing the bin centers for the wind speed and wind direction 
        data, limit_month: optional list of months (1,2,...,12) for limiting data, st_date, 
        en_date: optional start and end dates for limiting the date range of the obtained data 
        (str: 'MM-DD-YYYY')  

        """

        # check inputs
        if st_date is not None:
            if dateutil.parser.parse(st_date) > dateutil.parser.parse('12-13-2013 23:00'):
                print('Error, invalid date range. Valid range: 01-01-2007 - 12/31/2013')
                return None
        if en_date is not None:
            if dateutil.parser.parse(en_date)  < dateutil.parser.parse('01-01-2007 00:00'):
                print('Error, invalid date range. Valid range: 01-01-2007 - 12/31/2013')
                return None

        if ht not in [10,40,60,80,100,120,140,160,200]:
            print('Error, invalid height. Valid heights: 10, 40, 60, 80, 100, 120, 140, 160, 200')
            return None


        # load wind speeds and directions from WIND Toolkit 
        df = self.load_wind_toolkit_hsds(lat, lon, ht = ht, limit_month=limit_month, st_date=st_date, en_date=en_date)

        # check for errors in loading wind toolkit data
        if df is None:
            return None

        # Start by simply round and wrapping the wind direction and wind speed columns
        df['wd'] = geo.wrap_360(df.wd.round())
        df['ws'] = geo.wrap_360(df.ws.round())

        # Now group up
        df['freq_val'] = 1.
        df = df.groupby(['ws','wd']).sum()
        df['freq_val'] = df.freq_val.astype(float) / df.freq_val.sum()
        df = df.reset_index()

        # Save the df at this point
        self.df = df

        # Resample onto the provided wind speed and wind direction binnings
        self.internal_resample_wind_speed(ws=ws)
        self.internal_resample_wind_direction(wd=wd)

        return df

    def load_wind_toolkit_hsds(self, lat, lon, ht = 100, limit_month=None, st_date=None, en_date=None):
        """ 
        Given a lat/long coordinate in the continental US return a dataframe containing wind speeds 
        and wind directions at the specified height for each hour in the date range given. The wind data 
        is obtained from the WIND Toolkit dataset (https://www.nrel.gov/grid/wind-toolkit.html) using 
        the HSDS service (see https://github.com/NREL/hsds-examples). The wind data returned is obtained 
        from the nearest 2km x 2km grid point to the input coordinate.

        input: lat, lon: coordinates in degrees, ht: height above ground where wind information is 
        obtained (m), limit_month: optional list of months (1,2,...,12) for limiting data, st_date, 
        en_date: optional start and end dates for limiting the date range of the obtained data 
        (str: 'MM-DD-YYYY')  

        output: dataframe with wind speed and wind direction columns containing hourly data 

        """

        # Open the wind data "file"
        # server endpoint, username, password is found via a config file
        f = h5pyd.File("/nrel/wtk-us.h5", 'r')

        # assign wind direction, wind speed, and time datasets for the desired height
        wd_dset = f['winddirection_'+str(ht)+'m']
        ws_dset = f['windspeed_'+str(ht)+'m']
        dt = f['datetime']
        dt = pd.DataFrame({'datetime': dt[:]},index=range(0,dt.shape[0]))
        dt['datetime'] = dt['datetime'].apply(dateutil.parser.parse)

        # find dataset indices from lat/long
        Location_idx = self.indices_for_coord( f, lat, lon )

        # check if in bounds
        if (Location_idx[0] < 0) | (Location_idx[0] >= wd_dset.shape[1]) | (Location_idx[1] < 0) | (Location_idx[1] >= wd_dset.shape[2]):
            print('Error, coordinates out of bounds. WIND Toolkit database covers the continental United States.')
            return None

        # create dataframe with wind direction and wind speed
        df = pd.DataFrame()
        df['wd'] = wd_dset[:,Location_idx[0],Location_idx[1]]
        df['ws'] = ws_dset[:,Location_idx[0],Location_idx[1]]
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
        df = df[['wd','ws']]

        return df

    def indices_for_coord(self, f, lat_index, lon_index):
        # Function obtained from: https://github.com/NREL/hsds-examples/blob/master/notebooks/01_introduction.ipynb "indicesForCoord"
        # This function finds the nearest x/y indices for a given lat/lon.
        # Rather than fetching the entire coordinates database, which is 500+ MB, this
        # uses the Proj4 library to find a nearby point and then converts to x/y indices

        dset_coords = f['coordinates']
        projstring = """+proj=lcc +lat_1=30 +lat_2=60 
                        +lat_0=38.47240422490422 +lon_0=-96.0 
                        +x_0=0 +y_0=0 +ellps=sphere 
                        +units=m +no_defs """
        projectLcc = Proj(projstring)
        origin_ll = reversed(dset_coords[0][0])  # Grab origin directly from database
        origin = projectLcc(*origin_ll)
        
        coords = (lon_index,lat_index)
        coords = projectLcc(*coords)
        delta = np.subtract(coords, origin)
        ij = [int(round(x/2000)) for x in delta]
        return tuple(reversed(ij))

    def plot_wind_speed_all(self,ax=None):

        if ax is None:
            _, ax = plt.subplots()

        df_plot = self.df.groupby('ws').sum()
        ax.plot(self.ws, df_plot.freq_val)


    def plot_wind_speed_by_direction(self,dirs,ax=None):

        # Get a downsampled frame
        df_plot = self.resample_wind_direction(self.df,wd=dirs)

        if ax is None:
            _, ax = plt.subplots()

        for wd in dirs:
            df_plot_sub = df_plot[df_plot.wd==wd]
            ax.plot(df_plot_sub.ws, df_plot_sub['freq_val'],label=wd)
        ax.legend()

    def plot_wind_rose(self, ax=None, color_map='viridis_r',ws_right_edges=np.array([5,10,15,20,25]),wd_bins = np.arange(0,360,15.)):

        # Based on code provided by Patrick Murphy
        
        # Resample data onto bins
        # df_plot = self.resample_wind_speed(self.df,ws=ws_bins)
        df_plot = self.resample_wind_direction(self.df,wd=wd_bins)

        # Make labels for wind speed based on edges
        ws_step = ws_right_edges[1] - ws_right_edges[0]
        # ws = ws_edges
        # ws_edges = (ws - ws_step / 2.0)
        # ws_edges = np.append(ws_edges,np.array(ws[-1] + ws_step / 2.0))
        # ws_edges = np.append([0], ws_right_edges)
        ws_labels = ['%d-%d m/s'  % (w - ws_step,w) for w in ws_right_edges]

        # Grab the wd_step
        wd_step = wd_bins[1] - wd_bins[0]

        # Set up figure
        if ax is None:
            _, ax = plt.subplots(subplot_kw=dict(polar=True))

        # Get a color array
        color_array = cm.get_cmap(color_map,len(ws_right_edges))

        for wd_idx, wd in enumerate(wd_bins):
            rects = list()
            df_plot_sub = df_plot[df_plot.wd==wd]
            for ws_idx, ws in enumerate(ws_right_edges[::-1]):  
                plot_val = df_plot_sub[df_plot_sub.ws<=ws].freq_val.sum() # Get the sum of frequency up to this wind speed
                rects.append(ax.bar(np.radians(wd), plot_val, width = 0.9 * np.radians(wd_step), color = color_array(ws_idx),edgecolor='k'))
            # break

        # Configure the plot
        ax.legend(reversed(rects),ws_labels)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi/2.0)
        ax.set_theta_zero_location("N")
        ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

        return ax

    def export_for_floris_opt(self):

        # Return a list of tuples, where each tuple is (ws,wd,freq)
        return [tuple(x) for x in self.df.values]