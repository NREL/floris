# NREL 2019
# Patrick Duffy

import matplotlib.pyplot as plt
import floris.tools as wfct
import floris.tools.visualization as vis
from floris.tools.optimization import YawOptimizationWindRose
import floris.tools.wind_rose as rose
import floris.tools.power_rose as pr
import numpy as np
import pandas as pd
from math import sqrt, floor
import time


class aep_calc():

    def __init__(self):
        """Initialize AEP calculation

        AEP calculation class. A container for the single_aep_value() method.

        """

    def single_aep_value(self, lat, lon, wt_cap_MW, n_wt, plot_layout=False,
                         wake_model='gauss', fwrop=None, plot_rose=False,
                         grid_spc=7):
        """Single AEP value

        Calculates AEP for single set of input parameters. Returns waked and
        wake-free AEP, as well as the percent wake loss.

        Inputs
        ----------
        lat : latitude coordinate in decimal degrees
        lon : longitude coordinate in decimal degrees
        wt_cap_MW  : Individual wind turbine capacity in MW
        n_wt : Number of wind turbines
        plot_layout : Plots the wind farm layout when true. Default: False
        wake_model : Wake model used in the calculation. Default: gauss
        fwrop : Floris wind rose object path. Default: None (uses Wind Toolkit)
        plot_rose : Plots wind rose used in the AEP calculation. Default: False
        grd_spc : turbine spacing in grid layout in # rotor diams. Default: 7

        (Still want to implement these as inputs):
            wts : wind turbines in the wind farm, and their respective properties
                want to check a library for a given rated power value and load
                if a turbine model is not in the library, raise an exception
            wake_combin : wake summation method for multiple wakes

        Returns
        -------
        wake_free_aep_GWh : AEP ignoring wake effects in GWh
        aep_GWh : AEP including wake effects in GWh
        percent_loss : Percent of wake_free_aep lost due to wakes
        """

        # Setup floris object - might want to split this up later when have WT library
        fi = wfct.floris_utilities.FlorisInterface('example_input.json')

        # Make a grid layout
        D = fi.floris.farm.turbines[0].rotor_diameter
        layout_x, layout_y =self.make_grid_layout(wt_cap_MW, n_wt, D,
                                                  grid_spc=grid_spc,
                                                  plant_cap_MW=600)

        # Set up the model and update the floris object
        fi.floris.farm.set_wake_model(wake_model)
        fi.reinitialize_flow_field(layout_array=(layout_x, layout_y),
                                   wind_direction=270.0,wind_speed=8.0)
        fi.calculate_wake()

        # Plot wf layout
        if plot_layout:
            fig, ax = plt.subplots()
            vis.plot_turbines(ax, layout_x, layout_y,
                              yaw_angles=np.zeros(len(layout_x)), D=D)
            ax.set_title('Wind Farm Layout')

        # Wind data (either from wind rose object, or from windtoolkit)
        wind_rose = rose.WindRose()
        if fwrop is None:
            print('Accessing Wind Toolkit...')
            # fetch the data from wind tookit
            wd_list = np.arange(0,360,5) # 5 degree wd bins
            ws_list = np.arange(0,26,1)  # 1 m/s ws bins

            df = wind_rose.import_from_wind_toolkit_hsds(lat,
                                                         lon,
                                                         ht = 100,
                                                         wd = wd_list,
                                                         ws = ws_list,
                                                         limit_month = None,
                                                         st_date = None,
                                                         en_date = None)
        else:
            df = wind_rose.load(fwrop)

        if plot_rose:
            wind_rose.plot_wind_rose()

        # Instantiate the Optimization object
        yaw_opt = YawOptimizationWindRose(fi, df.wd, df.ws,
                                       minimum_yaw_angle=0,
                                       maximum_yaw_angle=0,
                                       minimum_ws=4.0,
                                       maximum_ws=25.0)

        # Determine baseline power with and without wakes
        df_base = yaw_opt.calc_baseline_power()


        # Combine wind farm-level power into one dataframe
        df_power = pd.DataFrame({'ws':df.ws,'wd':df.wd, \
            'freq_val':df.freq_val,'power_no_wake':df_base.power_no_wake, \
            'power_baseline':df_base.power_baseline})

        # Set up the power rose
        df_turbine_power_no_wake = pd.DataFrame([list(row) for row in df_base['turbine_power_no_wake']],
                                                 columns=[str(i) for i in range(1,n_wt+1)])
        df_turbine_power_no_wake['ws'] = df.ws
        df_turbine_power_no_wake['wd'] = df.wd
        df_turbine_power_baseline = pd.DataFrame([list(row) for row in df_base['turbine_power_baseline']],
                                                  columns=[str(i) for i in range(1,n_wt+1)])
        df_turbine_power_baseline['ws'] = df.ws
        df_turbine_power_baseline['wd'] = df.wd
        case_name = 'Wind Farm'
        power_rose = pr.PowerRose(case_name, df_power, df_turbine_power_no_wake,
                                  df_turbine_power_baseline)

        # Values to return
        wake_free_aep_GWh = power_rose.total_no_wake
        aep_GWh = power_rose.total_baseline
        percent_loss = 100*power_rose.baseline_wake_loss

        return aep_GWh, wake_free_aep_GWh, percent_loss


    def isPerfect(self, N):
        """Function to check if a number is perfect square or not

        taken from:
        https://www.geeksforgeeks.org/closest-perfect-square-and-its-distance/
        by sahishelangia
        """
        if (sqrt(N) - floor(sqrt(N)) != 0):
            return False
        return True


    def getClosestPerfectSquare(self, N):
        """Function to find the closest perfect square taking minimum steps to
            reach from a number

        taken from:
        https://www.geeksforgeeks.org/closest-perfect-square-and-its-distance/
        by sahishelangia
        """
        if (self.isPerfect(N)):
            distance = 0
            return N, distance

        # Variables to store first perfect square number above and below N
        aboveN = -1
        belowN = -1
        n1 = 0

        # Finding first perfect square number greater than N
        n1 = N + 1
        while (True):
            if (self.isPerfect(n1)):
                aboveN = n1
                break
            else:
                n1 += 1

        # Finding first perfect square number less than N
        n1 = N - 1
        while (True):
            if (self.isPerfect(n1)):
                belowN = n1
                break
            else:
                n1 -= 1

        # Variables to store the differences
        diff1 = aboveN - N
        diff2 = N - belowN

        if (diff1 > diff2):
            return belowN, -diff2
        else:
            return aboveN, diff1


    def make_grid_layout(self, wt_cap_MW, n_wt, D, grid_spc, plant_cap_MW=600):
        """Make a grid layout (close as possible to a square grid)

        Inputs:
        -------
            wt_cap_MW : float
                Wind turbine capacity in MW
            n_wt : float
                Number of wind turbines in the plant
            D : float (or might want array_like if diff wt models are used)
                Wind turbine rotor diameter(s) in meters
            grid_spc : float
                Spacing between rows and columns in number of rotor diams D
            plant_cap_MW : float
                Total wind plant capacity in MW

        Returns:
        --------
            layout_x : array_like
                X positions of the wind turbines in the plant
            layout_y : array_like
                Y positions of the wind turbines in the plant
        """
        # Check the plant capacity and number of turbines
        if plant_cap_MW/wt_cap_MW != n_wt:
            # Print a warning
            print('Note n_wt*wt_cap may not agree with total plant capacity.')

        # Initialize layout variables
        layout_x = []
        layout_y = []

        # Find the closest square root
        close_square, dist = self.getClosestPerfectSquare(n_wt)
        side_length = int(sqrt(close_square))

        # Build a square grid
        for i in range(side_length):
            for k in range(side_length):
                layout_x.append(i*grid_spc*D)
                layout_y.append(k*grid_spc*D)

        # Check dist and determine what to do
        if dist == 0:
            # do nothing
            pass
        elif dist > 0:
            # square>n_wt : remove locations
            del(layout_x[close_square-dist:close_square])
            del(layout_y[close_square-dist:close_square])
            # maybe convert format and transpose
        else:
            # square < n_w_t : add a partial row
            for i in range(abs(dist)):
                layout_x.append(sqrt(close_square)*grid_spc*D)
                layout_y.append(i*grid_spc*D)

        return layout_x, layout_y



def main():
    # location and wind rose data
    # center usa:
    lat = 39.8283
    lon = -98.5795
    wind_data = 'windtoolkit_geo_center_us.p'
    # california offshore:
    #lat = 35.236497
    #lon= -120.991624
    #wind_data = None

    wt_cap_MW = 5    # Turbine rating
    n_wt = 4         # Number of turbines

    # possible wake models: 'curl', 'gauss', 'jensen', 'multizone'
    aep=aep_calc()
    start = time.time()
    aep_GWh,wake_free_aep_GWh,percent_loss = aep.single_aep_value(lat,
                                                                  lon,
                                                                  wt_cap_MW,
                                                                  n_wt,
                                                                  plot_layout = False,
                                                                  wake_model='gauss',
                                                                  fwrop=wind_data,
                                                                  plot_rose = False)
    end = time.time()
    print('Time',end-start)
    print('AEP [GWh]=', aep_GWh, 'Wake free AEP [GWh]=',wake_free_aep_GWh, 'Wake loss [%]=',percent_loss)

main()
