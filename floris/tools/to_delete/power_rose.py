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

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from floris.utilities import wrap_180


# TODO: organize by private and public methods


class PowerRose:
    """
    The PowerRose class is used to organize information about wind farm power
    production for different wind conditions (e.g., wind speed, wind direction)
    along with their frequencies of occurance to calculate the resulting annual
    energy production (AEP). Power production and AEP are considered for
    baseline operation, ideal operation without wake losses, and optionally
    optimal operation with wake steering. The primary purpose of the PowerRose
    class is for visualizing and reporting energy production and energy gains
    from wake steering. A PowerRose object can be populated with user-specified
    wind rose and power data (for example, using a :py:class:`~.tools
    WindRose` object) or data from a previously saved PowerRose object can be
    loaded.
    """

    def __init__(self,):
        """
        Instantiate a PowerRose object. No explicit arguments required, and an
        additional method will need to be called to populate the PowerRose
        object with data.
        """

    def load(self, filename):
        """
        This method loads data from a previously saved PowerRose pickle file
        into a PowerRose object.

        Args:
            filename (str): Path and filename of pickle file to load.
        """

        (
            self.name,
            self.df_windrose,
            self.power_no_wake,
            self.power_baseline,
            self.power_opt,
            self.use_opt,
        ) = pickle.load(open(filename, "rb"))

        # Compute energies
        self.df_power = pd.DataFrame(
            {"wd": self.df_windrose["wd"], "ws": self.df_windrose["ws"]}
        )
        self._compute_energy()

        # Compute totals
        self._compute_totals()

    def save(self, filename):
        """
        This method saves PowerRose data as a pickle file so that it can be
        imported into a PowerRose object later.

        Args:
            filename (str): Path and filename of pickle file to save.
        """
        pickle.dump(
            [
                self.name,
                self.df_windrose,
                self.power_no_wake,
                self.power_baseline,
                self.power_opt,
                self.use_opt,
            ],
            open(filename, "wb"),
        )

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
        print("Norming frequency total of %.2f to 1.0" % df.freq_val.sum())
        df["freq_val"] = df.freq_val / df.freq_val.sum()
        return df

    def _compute_energy(self):
        self.df_power["energy_no_wake"] = self.df_windrose.freq_val * self.power_no_wake
        self.df_power["energy_baseline"] = (
            self.df_windrose.freq_val * self.power_baseline
        )
        if self.use_opt:
            self.df_power["energy_opt"] = self.df_windrose.freq_val * self.power_opt

    def _compute_totals(self):
        df = self.df_power.copy(deep=True)
        df = df.sum()

        # Get total annual energy amounts
        self.total_no_wake = (8760 / 1e9) * df.energy_no_wake
        self.total_baseline = (8760 / 1e9) * df.energy_baseline
        if self.use_opt:
            self.total_opt = (8760 / 1e9) * df.energy_opt

        # Get wake loss amounts
        self.baseline_percent = self.total_baseline / self.total_no_wake
        self.baseline_wake_loss = 1 - self.baseline_percent

        if self.use_opt:
            self.opt_percent = self.total_opt / self.total_no_wake
            self.opt_wake_loss = 1 - self.opt_percent

        # Percent gain
        if self.use_opt:
            self.percent_gain = (
                self.total_opt - self.total_baseline
            ) / self.total_baseline
            self.reduction_in_wake_loss = (
                -1
                * (self.opt_wake_loss - self.baseline_wake_loss)
                / self.baseline_wake_loss
            )

    def make_power_rose_from_user_data(
        self, name, df_windrose, power_no_wake, power_baseline, power_opt=None
    ):
        """
        This method populates the PowerRose object with a user-specified wind
        rose containing wind direction, wind speed, and additional optional
        variables, as well as baseline wind farm power, ideal wind farm power
        without wake losses, and optionally optimal wind farm power with wake
        steering corresponding to each wind condition.

        TODO: Add inputs for turbine-level power and optimal yaw offsets.

        Args:
            name (str): The name of the PowerRose object.
            df_windrose (pandas.DataFrame): A DataFrame with wind rose
                information containing at least
                the following columns:

                - **wd** (*float*) - Wind direction bin center values (deg).
                - **ws** (*float*) - Wind speed bin center values (m/s).
                - **freq_val** (*float*) - The frequency of occurance of the
                wind conditions in the other columns.

            power_no_wake (iterable): A list of wind farm power without wake
                losses corresponding to the wind conditions in df_windrose (W).
            power_baseline (iterable): A list of baseline wind farm power with
                wake losses corresponding to the wind conditions in df_windrose
                (W).
            power_opt (iterable, optional): A list of optimal wind farm power
                with wake steering corresponding to the wind conditions in
                df_windrose (W). Defaults to None.
        """
        self.name = name
        if df_windrose is not None:
            self.df_windrose = self._norm_frequency(df_windrose)
        self.power_no_wake = power_no_wake
        self.power_baseline = power_baseline
        self.power_opt = power_opt

        # Only use_opt data if provided
        if power_opt is None:
            self.use_opt = False
        else:
            self.use_opt = True

        # # Make a single combined frame in case it's useful (Set aside for now)
        # self.df_combine = self._all_combine()

        # Compute energies
        self.df_power = pd.DataFrame({"wd": df_windrose["wd"], "ws": df_windrose["ws"]})
        self._compute_energy()

        # Compute totals
        self._compute_totals()

    def report(self):
        """
        This method prints information about annual energy production (AEP)
        using the PowerRose object data. The AEP in GWh is listed for ideal
        operation without wake losses, baseline operation, and optimal
        operation with wake steering, if optimal power data are stored. The
        wind farm efficiency (% of ideal energy production) and wake loss
        percentages are listed for baseline and optimal operation (if optimal
        power is stored), along with the AEP gain from wake steering (again, if
        optimal power is stored). The AEP gain from wake steering is also
        listed as a percentage of wake losses recovered, if applicable.
        """
        if self.use_opt:
            print("=============================================")
            print("Case %s has results:" % self.name)
            print("=============================================")
            print("-\tNo-Wake\t\tBaseline\tOpt ")
            print("---------------------------------------------")
            print(
                "AEP (GWh)\t%.1E\t\t%.1E\t\t%.1E"
                % (self.total_no_wake, self.total_baseline, self.total_opt)
            )
            print(
                "%%\t--\t\t%.1f%%\t\t%.1f%%"
                % (100.0 * self.baseline_percent, 100.0 * self.opt_percent)
            )
            print(
                "Wk Loss\t--\t\t%.1f%%\t\t%.1f%%"
                % (100.0 * self.baseline_wake_loss, 100.0 * self.opt_wake_loss)
            )
            print("AEP Gain --\t\t--\t\t%.1f%%" % (100.0 * self.percent_gain))
            print("Loss Red --\t\t--\t\t%.1f%%" % (100.0 * self.reduction_in_wake_loss))
        else:
            print("=============================================")
            print("Case %s has results:" % self.name)
            print("=============================================")
            print("-\tNo-Wake\t\tBaseline ")
            print("---------------------------------------------")
            print("AEP (GWh)\t%.1E\t\t%.1E" % (self.total_no_wake, self.total_baseline))
            print("%%\t--\t\t%.1f%%" % (100.0 * self.baseline_percent))
            print("Wk Loss\t--\t\t%.1f%%" % (100.0 * self.baseline_wake_loss))

    def plot_by_direction(self, axarr=None):
        """
        This method plots energy production, wind farm efficiency, and energy
        gains from wake steering (if applicable) as a function of wind
        direction. If axes are not provided, new ones are created. The plots
        include:

        1) The energy production as a function of wind direction for the
        baseline and, if applicable, optimal wake steering cases normalized by
        the maximum energy production.
        2) The wind farm efficiency (energy production relative to energy
        production without wake losses) as a function of wind direction for the
        baseline and, if applicable, optimal wake steering cases.
        3) Percent gain in energy production with optimal wake steering as a
        function of wind direction. This third plot is only created if optimal
        power data are stored in the PowerRose object.

        Args:
            axarr (numpy.ndarray, optional): An array of 2 or 3
                :py:class:`matplotlib.axes._subplots.AxesSubplot` axes objects
                on which data are plotted. Three axes are rquired if the
                PowerRose object contains optimal power data. Default is None.

        Returns:
            numpy.ndarray: An array of 2 or 3
            :py:class:`matplotlib.axes._subplots.AxesSubplot` axes objects on
            which the data are plotted.
        """

        df = self.df_power.copy(deep=True)
        df = df.groupby("wd").sum().reset_index()

        if self.use_opt:

            if axarr is None:
                fig, axarr = plt.subplots(3, 1, sharex=True)

            ax = axarr[0]
            ax.plot(
                df.wd,
                df.energy_baseline / np.max(df.energy_opt),
                label="Baseline",
                color="k",
            )
            ax.axhline(
                np.mean(df.energy_baseline / np.max(df.energy_opt)), color="r", ls="--"
            )
            ax.plot(
                df.wd,
                df.energy_opt / np.max(df.energy_opt),
                label="Optimized",
                color="r",
            )
            ax.axhline(
                np.mean(df.energy_opt / np.max(df.energy_opt)), color="r", ls="--"
            )
            ax.set_ylabel("Normalized Energy")
            ax.grid(True)
            ax.legend()
            ax.set_title(self.name)

            ax = axarr[1]
            ax.plot(
                df.wd,
                df.energy_baseline / df.energy_no_wake,
                label="Baseline",
                color="k",
            )
            ax.axhline(
                np.mean(df.energy_baseline) / np.mean(df.energy_no_wake),
                color="k",
                ls="--",
            )
            ax.plot(
                df.wd, df.energy_opt / df.energy_no_wake, label="Optimized", color="r"
            )
            ax.axhline(
                np.mean(df.energy_opt) / np.mean(df.energy_no_wake), color="r", ls="--"
            )
            ax.set_ylabel("Wind Farm Efficiency")
            ax.grid(True)
            ax.legend()

            ax = axarr[2]
            ax.plot(
                df.wd,
                100.0 * (df.energy_opt - df.energy_baseline) / df.energy_baseline,
                "r",
            )
            ax.axhline(
                100.0
                * (df.energy_opt.mean() - df.energy_baseline.mean())
                / df.energy_baseline.mean(),
                df.energy_baseline.mean(),
                color="r",
                ls="--",
            )
            ax.set_ylabel("Percent Gain")
            ax.set_xlabel("Wind Direction (deg)")

            return axarr

        else:

            if axarr is None:
                fig, axarr = plt.subplots(2, 1, sharex=True)

            ax = axarr[0]
            ax.plot(
                df.wd,
                df.energy_baseline / np.max(df.energy_baseline),
                label="Baseline",
                color="k",
            )
            ax.axhline(
                np.mean(df.energy_baseline / np.max(df.energy_baseline)),
                color="r",
                ls="--",
            )
            ax.set_ylabel("Normalized Energy")
            ax.grid(True)
            ax.legend()
            ax.set_title(self.name)

            ax = axarr[1]
            ax.plot(
                df.wd,
                df.energy_baseline / df.energy_no_wake,
                label="Baseline",
                color="k",
            )
            ax.axhline(
                np.mean(df.energy_baseline) / np.mean(df.energy_no_wake),
                color="k",
                ls="--",
            )
            ax.set_ylabel("Wind Farm Efficiency")
            ax.grid(True)
            ax.legend()

            ax.set_xlabel("Wind Direction (deg)")

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
