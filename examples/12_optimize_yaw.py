# Copyright 2022 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


from time import perf_counter as timerpc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from floris.tools import FlorisInterface
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR


"""
This example demonstrates how to perform a yaw optimization and evaluate the performance
over a full wind rose.

The beginning of the file contains the definition of several functions used in the main part
of the script.

Within the main part of the script, we first load the wind rose information. We then initialize
our Floris Interface object. We determine the baseline AEP using the wind rose information, and
then perform the yaw optimization over 72 wind directions with 1 wind speed per direction. The
optimal yaw angles are then used to determine yaw angles across all the wind speeds included in
the wind rose. Lastly, the final AEP is calculated and analysis of the results are
shown in several plots.
"""

def load_floris():
    # Load the default example floris object
    fi = FlorisInterface("inputs/gch.yaml") # GCH model matched to the default "legacy_gauss" of V2
    # fi = FlorisInterface("inputs/cc.yaml") # New CumulativeCurl model

    # Specify wind farm layout and update in the floris object
    N = 5  # number of turbines per row and per column
    X, Y = np.meshgrid(
        5.0 * fi.floris.farm.rotor_diameters_sorted[0][0][0] * np.arange(0, N, 1),
        5.0 * fi.floris.farm.rotor_diameters_sorted[0][0][0] * np.arange(0, N, 1),
    )
    fi.reinitialize(layout_x=X.flatten(), layout_y=Y.flatten())

    return fi


def load_windrose():
    fn = "inputs/wind_rose.csv"
    df = pd.read_csv(fn)
    df = df[(df["ws"] < 22)].reset_index(drop=True)  # Reduce size
    df["freq_val"] = df["freq_val"] / df["freq_val"].sum() # Normalize wind rose frequencies

    return df


def calculate_aep(fi, df_windrose, column_name="farm_power"):
    from scipy.interpolate import NearestNDInterpolator

    # Define columns
    nturbs = len(fi.layout_x)
    yaw_cols = ["yaw_{:03d}".format(ti) for ti in range(nturbs)]

    if "yaw_000" not in df_windrose.columns:
        df_windrose[yaw_cols] = 0.0  # Add zeros

    # Derive the wind directions and speeds we need to evaluate in FLORIS
    wd_array = np.array(df_windrose["wd"].unique(), dtype=float)
    ws_array = np.array(df_windrose["ws"].unique(), dtype=float)
    yaw_angles = np.array(df_windrose[yaw_cols], dtype=float)
    fi.reinitialize(wind_directions=wd_array, wind_speeds=ws_array)

    # Map angles from dataframe onto floris wind direction/speed grid
    X, Y = np.meshgrid(wd_array, ws_array, indexing='ij')
    interpolant = NearestNDInterpolator(df_windrose[["wd", "ws"]], yaw_angles)
    yaw_angles_floris = interpolant(X, Y)

    # Calculate FLORIS for every WD and WS combination and get the farm power
    fi.calculate_wake(yaw_angles_floris)
    farm_power_array = fi.get_farm_power()

    # Now map FLORIS solutions to dataframe
    interpolant = NearestNDInterpolator(
        np.vstack([X.flatten(), Y.flatten()]).T,
        farm_power_array.flatten()
    )
    df_windrose[column_name] = interpolant(df_windrose[["wd", "ws"]])  # Save to dataframe
    df_windrose[column_name] = df_windrose[column_name].fillna(0.0)  # Replace NaNs with 0.0

    # Calculate AEP in GWh
    aep = np.dot(df_windrose["freq_val"], df_windrose[column_name]) * 365 * 24 / 1e9

    return aep


if __name__ == "__main__":
    # Load a dataframe containing the wind rose information
    df_windrose = load_windrose()

    # Load FLORIS
    fi = load_floris()
    fi.reinitialize(wind_speeds=8.0)
    nturbs = len(fi.layout_x)

    # First, get baseline AEP, without wake steering
    start_time = timerpc()
    print(" ")
    print("===========================================================")
    print("Calculating baseline annual energy production (AEP)...")
    aep_bl = calculate_aep(fi, df_windrose, "farm_power_baseline")
    t = timerpc() - start_time
    print("Baseline AEP: {:.3f} GWh. Time spent: {:.1f} s.".format(aep_bl, t))
    print("===========================================================")
    print(" ")

    # Now optimize the yaw angles using the Serial Refine method
    print("Now starting yaw optimization for the entire wind rose...")
    start_time = timerpc()
    fi.reinitialize(
        wind_directions=np.arange(0.0, 360.0, 5.0),
        wind_speeds=[8.0]
    )
    yaw_opt = YawOptimizationSR(
        fi=fi,
        minimum_yaw_angle=0.0,  # Allowable yaw angles lower bound
        maximum_yaw_angle=20.0,  # Allowable yaw angles upper bound
        Ny_passes=[5, 4],
        exclude_downstream_turbines=True,
        exploit_layout_symmetry=True,
    )

    df_opt = yaw_opt.optimize()
    end_time = timerpc()
    t_tot = end_time - start_time
    t_fi = yaw_opt.time_spent_in_floris

    print("Optimization finished in {:.2f} seconds.".format(t_tot))
    print(" ")
    print(df_opt)
    print(" ")

    # Now define how the optimal yaw angles for 8 m/s are applied over the other wind speeds
    yaw_angles_opt = np.vstack(df_opt["yaw_angles_opt"])
    yaw_angles_wind_rose = np.zeros((df_windrose.shape[0], nturbs))
    for ii, idx in enumerate(df_windrose.index):
        wind_speed = df_windrose.loc[idx, "ws"]
        wind_direction = df_windrose.loc[idx, "wd"]

        # Interpolate the optimal yaw angles for this wind direction from df_opt
        id_opt = df_opt["wind_direction"] == wind_direction
        yaw_opt_full = np.array(df_opt.loc[id_opt, "yaw_angles_opt"])[0]

        # Now decide what to do for different wind speeds
        if (wind_speed < 4.0) | (wind_speed > 14.0):
            yaw_opt = np.zeros(nturbs)  # do nothing for very low/high speeds
        elif wind_speed < 6.0:
            yaw_opt = yaw_opt_full * (6.0 - wind_speed) / 2.0  # Linear ramp up
        elif wind_speed > 12.0:
            yaw_opt = yaw_opt_full * (14.0 - wind_speed) / 2.0  # Linear ramp down
        else:
            yaw_opt = yaw_opt_full  # Apply full offsets between 6.0 and 12.0 m/s

        # Save to collective array
        yaw_angles_wind_rose[ii, :] = yaw_opt

    # Add optimal and interpolated angles to the wind rose dataframe
    yaw_cols = ["yaw_{:03d}".format(ti) for ti in range(nturbs)]
    df_windrose[yaw_cols] = yaw_angles_wind_rose

    # Now get AEP with optimized yaw angles
    start_time = timerpc()
    print("==================================================================")
    print("Calculating annual energy production (AEP) with wake steering...")
    aep_opt = calculate_aep(fi, df_windrose, "farm_power_opt")
    aep_uplift = 100.0 * (aep_opt / aep_bl - 1)
    t = timerpc() - start_time
    print("Optimal AEP: {:.3f} GWh. Time spent: {:.1f} s.".format(aep_opt, t))
    print("Relative AEP uplift by wake steering: {:.3f} %.".format(aep_uplift))
    print("==================================================================")
    print(" ")

    # Now calculate helpful variables and then plot wind rose information
    df = df_windrose.copy()
    df["farm_power_relative"] = (
        df["farm_power_opt"] / df["farm_power_baseline"]
    )
    df["farm_energy_baseline"] = df["freq_val"] * df["farm_power_baseline"]
    df["farm_energy_opt"] = df["freq_val"] * df["farm_power_opt"]
    df["energy_uplift"] = df["farm_energy_opt"] - df["farm_energy_baseline"]
    df["rel_energy_uplift"] = df["energy_uplift"] / df["energy_uplift"].sum()

    # Plot power and AEP uplift across wind direction
    fig, ax = plt.subplots(nrows=3, sharex=True)

    df_8ms = df[df["ws"] == 8.0].reset_index(drop=True)
    pow_uplift = 100 * (
        df_8ms["farm_power_opt"] / df_8ms["farm_power_baseline"] - 1
    )
    ax[0].bar(
        x=df_8ms["wd"],
        height=pow_uplift,
        color="darkgray",
        edgecolor="black",
        width=4.5,
    )
    ax[0].set_ylabel("Power uplift \n at 8 m/s (%)")
    ax[0].grid(True)

    dist = df.groupby("wd").sum().reset_index()
    ax[1].bar(
        x=dist["wd"],
        height=100 * dist["rel_energy_uplift"],
        color="darkgray",
        edgecolor="black",
        width=4.5,
    )
    ax[1].set_ylabel("Contribution to \n AEP uplift (%)")
    ax[1].grid(True)

    ax[2].bar(
        x=dist["wd"],
        height=dist["freq_val"],
        color="darkgray",
        edgecolor="black",
        width=4.5,
    )
    ax[2].set_xlabel("Wind direction (deg)")
    ax[2].set_ylabel("Frequency of \n occurrence (-)")
    ax[2].grid(True)
    plt.tight_layout()

    # Plot power and AEP uplift across wind direction
    fig, ax = plt.subplots(nrows=3, sharex=True)

    df_avg = df.groupby("ws").mean().reset_index(drop=False)
    mean_power_uplift = 100.0 * (df_avg["farm_power_relative"] - 1.0)
    ax[0].bar(
        x=df_avg["ws"],
        height=mean_power_uplift,
        color="darkgray",
        edgecolor="black",
        width=0.95,
    )
    ax[0].set_ylabel("Mean power \n uplift (%)")
    ax[0].grid(True)

    dist = df.groupby("ws").sum().reset_index()
    ax[1].bar(
        x=dist["ws"],
        height=100 * dist["rel_energy_uplift"],
        color="darkgray",
        edgecolor="black",
        width=0.95,
    )
    ax[1].set_ylabel("Contribution to \n AEP uplift (%)")
    ax[1].grid(True)

    ax[2].bar(
        x=dist["ws"],
        height=dist["freq_val"],
        color="darkgray",
        edgecolor="black",
        width=0.95,
    )
    ax[2].set_xlabel("Wind speed (m/s)")
    ax[2].set_ylabel("Frequency of \n occurrence (-)")
    ax[2].grid(True)
    plt.tight_layout()

    # Now plot yaw angle distributions over wind direction up to first three turbines
    for ti in range(np.min([nturbs, 3])):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(
            df_opt["wind_direction"],
            yaw_angles_opt[:, ti],
            "-o",
            color="maroon",
            markersize=3,
            label="For wind speeds between 6 and 12 m/s",
        )
        ax.plot(
            df_opt["wind_direction"],
            0.5 * yaw_angles_opt[:, ti],
            "-v",
            color="dodgerblue",
            markersize=3,
            label="For wind speeds of 5 and 13 m/s",
        )
        ax.plot(
            df_opt["wind_direction"],
            0.0 * yaw_angles_opt[:, ti],
            "-o",
            color="grey",
            markersize=3,
            label="For wind speeds below 4 and above 14 m/s",
        )
        ax.set_ylabel("Assigned yaw offsets (deg)")
        ax.set_xlabel("Wind direction (deg)")
        ax.set_title("Turbine {:d}".format(ti))
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

    plt.show()
