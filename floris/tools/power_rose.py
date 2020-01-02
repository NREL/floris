# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from floris.utilities import wrap_180


class PowerRose():
    """
    PowerRose object class used to parse data and generate figures.
    """

    def __init__(self, name, df_power, df_turbine_power_no_wake,
                   df_turbine_power_baseline, df_yaw=None, df_turbine_power_opt=None):

        """
        Additional initialization of the PowerRose object.

        Args:
            name (str): name of PowerRose object.
            df_power (pd.DataFrame): plant power data.
            
            df_turbine_power_no_wake (pd.DataFrame): wind turbine power
                data without wake losses.
            df_turbine_power_baseline (pd.DataFrame): wind turbine
                baseline power data.

            Only used in yaw optimization cases    
            df_yaw (pd.DataFrame): yaw data.
            df_turbine_power_opt (pd.DataFrame): optimal wind turbine
                power data.

        """
        self.name = name
        self.df_power = self._norm_frequency(df_power)
        self.df_yaw = df_yaw
        self.df_turbine_power_no_wake = df_turbine_power_no_wake
        self.df_turbine_power_baseline = df_turbine_power_baseline
        self.df_turbine_power_opt = df_turbine_power_opt

        # Only use_opt data if provided
        if (df_yaw is None) or (df_turbine_power_opt is None):
            self.use_opt = False
        else:
            self.use_opt = True

        # # Make a single combined frame in case it's useful (Set aside for now)
        # self.df_combine = self._all_combine()

        # Compute energies
        self._compute_energy()

        # Compute totals
        self._compute_totals()

    # def _all_combine(self):
    #     df_power = self.df_power.copy(deep=True)
    #     df_yaw = self.df_yaw.copy(deep=True)
    #     df_turbine_power_no_wake = self.df_turbine_power_no_wake.copy(
    #         deep=True)
    #     df_turbine_power_baseline = self.df_turbine_power_baseline.copy(
    #         deep=True)
    #     df_turbine_power_opt = self.df_turbine_power_opt.copy(deep=True)

    #     # Adjust the column names for uniqunes
    #     df_yaw.columns = [
    #         'yaw_%d' % c if type(c) is int else c for c in df_yaw.columns
    #     ]
    #     df_turbine_power_no_wake.columns = [
    #         'tnw_%d' % c if type(c) is int else c
    #         for c in df_turbine_power_no_wake.columns
    #     ]
    #     df_turbine_power_baseline.columns = [
    #         'tb_%d' % c if type(c) is int else c
    #         for c in df_turbine_power_baseline.columns
    #     ]
    #     df_turbine_power_opt.columns = [
    #         'topt_%d' % c if type(c) is int else c
    #         for c in df_turbine_power_opt.columns
    #     ]

    #     # Merge
    #     df_combine = df_power.merge(df_yaw, on=['ws', 'wd'])
    #     df_combine = df_combine.merge(df_turbine_power_no_wake,
    #                                   on=['ws', 'wd'])
    #     df_combine = df_combine.merge(df_turbine_power_baseline,
    #                                   on=['ws', 'wd'])
    #     df_combine = df_combine.merge(df_turbine_power_opt, on=['ws', 'wd'])

    #     return df_combine

    def _norm_frequency(self, df):
        print('Norming frequency total of %.2f to 1.0' % df.freq_val.sum())
        df['freq_val'] = df.freq_val / df.freq_val.sum()
        return df

    def _compute_energy(self):
        self.df_power[
            'energy_no_wake'] = self.df_power.freq_val * self.df_power.power_no_wake
        self.df_power[
            'energy_baseline'] = self.df_power.freq_val * self.df_power.power_baseline
        if self.use_opt:
            self.df_power[
                'energy_opt'] = self.df_power.freq_val * self.df_power.power_opt

    def _compute_totals(self):
        df = self.df_power.copy(deep=True)
        df = df.sum()

        # Get total annual energy amounts
        self.total_no_wake = (8760/1E9)*df.energy_no_wake
        self.total_baseline = (8760/1E9)*df.energy_baseline
        if self.use_opt:
            self.total_opt = (8760/1E9)*df.energy_opt

        # Get wake loss amounts
        self.baseline_percent = self.total_baseline / self.total_no_wake
        self.baseline_wake_loss = 1 - self.baseline_percent
        
        if self.use_opt:
            self.opt_percent = self.total_opt / self.total_no_wake
            self.opt_wake_loss = 1 - self.opt_percent

        # Percent gain
        if self.use_opt:
            self.percent_gain = (self.total_opt -
                                self.total_baseline) / self.total_baseline
            self.reduction_in_wake_loss = -1 * (
                self.opt_wake_loss -
                self.baseline_wake_loss) / self.baseline_wake_loss

    def report(self):
        """
        Print operational data information to screen.
        """
        if self.use_opt:
            print('=============================================')
            print('Case %s has results:' % self.name)
            print('=============================================')
            print('-\tNo-Wake\t\tBaseline\tOpt ')
            print('---------------------------------------------')
            print('AEP (GWh)\t%.1E\t\t%.1E\t\t%.1E' %
                (self.total_no_wake, self.total_baseline, self.total_opt))
            print('%%\t--\t\t%.1f%%\t\t%.1f%%' %
                (100. * self.baseline_percent, 100. * self.opt_percent))
            print('Wk Loss\t--\t\t%.1f%%\t\t%.1f%%' %
                (100. * self.baseline_wake_loss, 100. * self.opt_wake_loss))
            print('AEP Gain --\t\t--\t\t%.1f%%' % (100. * self.percent_gain))
            print('Loss Red --\t\t--\t\t%.1f%%' %
                (100. * self.reduction_in_wake_loss))
        else:
            print('=============================================')
            print('Case %s has results:' % self.name)
            print('=============================================')
            print('-\tNo-Wake\t\tBaseline ')
            print('---------------------------------------------')
            print('AEP (GWh)\t%.1E\t\t%.1E' %
                (self.total_no_wake, self.total_baseline))
            print('%%\t--\t\t%.1f%%' %
                (100. * self.baseline_percent))
            print('Wk Loss\t--\t\t%.1f%%' %
                (100. * self.baseline_wake_loss))


    def load(self, filename):
        """
        Load PowerRose object from pickle file.

        Args:
            filename (str): Read-from path to pickled PowerRose object.
        """

        # self.name, self.df_power, self.df_yaw, self.df_turbine_power_no_wake, self.df_turbine_power_baseline, self.df_turbine_power_opt, self.df_combine = pickle.load(
        #    open(filename, "rb"))

        self.name, self.df_power, self.df_yaw, self.df_turbine_power_no_wake, self.df_turbine_power_baseline, self.df_turbine_power_opt, self.use_opt = pickle.load(
            open(filename, "rb"))

        # Compute energies
        self._compute_energy()

        # Compute totals
        self._compute_totals()

    def save(self, filename):
        """
        Pickle a PowerRose Object

        Args:
            filename (str): Write-to path for PowerRose pickle.
        """
        pickle.dump([
            self.name, self.df_power, self.df_yaw,
            self.df_turbine_power_no_wake, self.df_turbine_power_baseline,
            self.df_turbine_power_opt, self.use_opt# , self.df_combine
        ], open(filename, "wb"))

    def plot_by_direction(self,axarr=None):
        """
        Bin energy data by direction and plot. Plots include: 
        1) The normalized energy production by wind direction for the baseline and 
        optimal wake steering cases. 
        2) The wind farm efficiency (energy production relative to energy production 
        without wake losses) by wind direction for the baseline and optimal wake steering cases. 
        3) Percent gain in energy production with optimal wake steering by wind direction.

        Args:
            axarr (numpy.ndarray of matplotlib.axes._subplots.AxesSubplot objects, optional): 
                array of 3 figure axes to which data should be plotted. Default is None. 

        Returns:
            axarr (numpy.ndarray of matplotlib.axes._subplots.AxesSubplot objects): 
                list of axis handles to which the data is plotted.
        """

        df = self.df_power.copy(deep=True)
        df = df.groupby('wd').sum().reset_index()


        if self.use_opt:

            if axarr is None:
                fig, axarr = plt.subplots(3, 1, sharex=True)

            ax = axarr[0]
            ax.plot(df.wd,
                    df.energy_baseline / np.max(df.energy_opt),
                    label='Baseline',
                    color='k')
            ax.axhline(np.mean(df.energy_baseline / np.max(df.energy_opt)),
                    color='r',
                    ls='--')
            ax.plot(df.wd,
                    df.energy_opt / np.max(df.energy_opt),
                    label='Optimized',
                    color='r')
            ax.axhline(np.mean(df.energy_opt / np.max(df.energy_opt)),
                    color='r',
                    ls='--')
            ax.set_ylabel('Normalized Energy')
            ax.grid(True)
            ax.legend()
            ax.set_title(self.name)

            ax = axarr[1]
            ax.plot(df.wd,
                    df.energy_baseline / df.energy_no_wake,
                    label='Baseline',
                    color='k')
            ax.axhline(np.mean(df.energy_baseline) / np.mean(df.energy_no_wake),
                    color='k',
                    ls='--')
            ax.plot(df.wd,
                    df.energy_opt / df.energy_no_wake,
                    label='Optimized',
                    color='r')
            ax.axhline(np.mean(df.energy_opt) / np.mean(df.energy_no_wake),
                    color='r',
                    ls='--')
            ax.set_ylabel('Wind Farm Efficiency')
            ax.grid(True)
            ax.legend()

            ax = axarr[2]
            ax.plot(
                df.wd,
                100. * (df.energy_opt - df.energy_baseline) / df.energy_baseline,
                'r')
            ax.axhline(100. * (df.energy_opt.mean() - df.energy_baseline.mean()) /
                    df.energy_baseline.mean(),
                    color='r',
                    ls='--')
            ax.set_ylabel('Percent Gain')
            ax.set_xlabel('Wind Direction (deg)')

            return axarr

        else:

            if axarr is None:
                fig, axarr = plt.subplots(2, 1, sharex=True)

            ax = axarr[0]
            ax.plot(df.wd,
                    df.energy_baseline / np.max(df.energy_baseline),
                    label='Baseline',
                    color='k')
            ax.axhline(np.mean(df.energy_baseline / np.max(df.energy_baseline)),
                    color='r',
                    ls='--')
            ax.set_ylabel('Normalized Energy')
            ax.grid(True)
            ax.legend()
            ax.set_title(self.name)

            ax = axarr[1]
            ax.plot(df.wd,
                    df.energy_baseline / df.energy_no_wake,
                    label='Baseline',
                    color='k')
            ax.axhline(np.mean(df.energy_baseline) / np.mean(df.energy_no_wake),
                    color='k',
                    ls='--')
            ax.set_ylabel('Wind Farm Efficiency')
            ax.grid(True)
            ax.legend()

            ax.set_xlabel('Wind Direction (deg)')

            return axarr

    # def wake_loss_at_direction(self, wd):
    #     """
    #     Calculate wake losses for a given direction. Plot rose figures
    #     for Power, Energy, Baseline power, Optimal gain, Total Gain,
    #     Percent Gain, etc.

    #     Args:
    #         wd (float): Wind direction of interest.

    #     Returns:
    #         tuple: tuple containing:

    #             -   **fig** (*plt.figure*): Figure handle.
    #             -   **axarr** (*list*): list of axis handles.
    #     """

    #     df = self.df_power.copy(deep=True)

    #     # Choose the nearest direction
    #     # Find nearest wind direction
    #     df['dist'] = np.abs(wrap_180(df.wd - wd))
    #     wd_select = df[df.dist == df.dist.min()]['wd'].unique()[0]
    #     print('Nearest wd to %.1f is %.1f' % (wd, wd_select))
    #     df = df[df.wd == wd_select]

    #     df = df.groupby('ws').sum().reset_index()

    #     fig, axarr = plt.subplots(4, 2, sharex=True, figsize=(14, 12))

    #     ax = axarr[0, 0]
    #     ax.set_title('Power')
    #     ax.plot(df.ws, df.power_no_wake, 'k', label='No Wake')
    #     ax.plot(df.ws, df.power_baseline, 'b', label='Baseline')
    #     ax.plot(df.ws, df.power_opt, 'r', label='Opt')
    #     ax.set_ylabel('Total')
    #     ax.grid()

    #     ax = axarr[0, 1]
    #     ax.set_title('Energy')
    #     ax.plot(df.ws, df.energy_no_wake, 'k', label='No Wake')
    #     ax.plot(df.ws, df.energy_baseline, 'b', label='Baseline')
    #     ax.plot(df.ws, df.energy_opt, 'r', label='Opt')
    #     ax.legend()
    #     ax.grid()

    #     ax = axarr[1, 0]
    #     ax.plot(df.ws,
    #             df.power_baseline / df.power_no_wake,
    #             'b',
    #             label='Baseline')
    #     ax.plot(df.ws, df.power_opt / df.power_no_wake, 'r', label='Opt')
    #     ax.set_ylabel('Percent')
    #     ax.grid()

    #     ax = axarr[1, 1]
    #     ax.plot(df.ws,
    #             df.energy_baseline / df.energy_no_wake,
    #             'b',
    #             label='Baseline')
    #     ax.plot(df.ws, df.energy_opt / df.energy_no_wake, 'r', label='Opt')
    #     ax.grid()

    #     ax = axarr[2, 0]
    #     ax.plot(df.ws, (df.power_opt - df.power_baseline), 'r')
    #     ax.set_ylabel('Total Gain')
    #     ax.grid()

    #     ax = axarr[2, 1]
    #     ax.plot(df.ws, (df.energy_opt - df.energy_baseline), 'r')
    #     ax.grid()

    #     ax = axarr[3, 0]
    #     ax.plot(df.ws, (df.power_opt - df.power_baseline) / df.power_baseline,
    #             'r')
    #     ax.set_ylabel('Percent Gain')
    #     ax.grid()
    #     ax.set_xlabel('Wind Speed (m/s)')

    #     ax = axarr[3, 1]
    #     ax.plot(df.ws,
    #             (df.energy_opt - df.energy_baseline) / df.energy_baseline, 'r')
    #     ax.grid()
    #     ax.set_xlabel('Wind Speed (m/s)')

    #     return fig, axarr