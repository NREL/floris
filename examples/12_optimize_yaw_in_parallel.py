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


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

from floris.tools import FlorisInterface, ParallelComputingInterface


"""
This example demonstrates how to perform a yaw optimization using parallel computing.
...
"""

def load_floris():
    # Load the default example floris object
    fi = FlorisInterface("inputs/gch.yaml") # GCH model matched to the default "legacy_gauss" of V2
    # fi = FlorisInterface("inputs/cc.yaml") # New CumulativeCurl model

    # Specify wind farm layout and update in the floris object
    N = 4  # number of turbines per row and per column
    X, Y = np.meshgrid(
        5.0 * fi.floris.farm.rotor_diameters_sorted[0][0][0] * np.arange(0, N, 1),
        5.0 * fi.floris.farm.rotor_diameters_sorted[0][0][0] * np.arange(0, N, 1),
    )
    fi.reinitialize(layout_x=X.flatten(), layout_y=Y.flatten())

    return fi


def load_windrose():
    # Grab a linear interpolant from this wind rose
    df = pd.read_csv("inputs/wind_rose.csv")
    interp = LinearNDInterpolator(points=df[["wd", "ws"]], values=df["freq_val"], fill_value=0.0)
    return df, interp


if __name__ == "__main__":
    # Parallel options
    max_workers = 16

    # Load a dataframe containing the wind rose information
    df_windrose, windrose_interpolant = load_windrose()

    # Load a FLORIS object for AEP calculations
    fi_aep = load_floris()
    wind_directions = np.arange(0.0, 360.0, 1.0)
    wind_speeds = np.arange(1.0, 25.0, 1.0)
    fi_aep.reinitialize(
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        turbulence_intensity=0.08  # Assume 8% turbulence intensity
    )

    # Pour this into a parallel computing interface
    parallel_interface = "concurrent"
    fi_aep_parallel = ParallelComputingInterface(
        fi=fi_aep,
        max_workers=max_workers,
        n_wind_direction_splits=max_workers,
        n_wind_speed_splits=1,
        interface=parallel_interface,
        print_timings=True,
    )

    # Calculate frequency of occurrence for each bin and normalize sum to 1.0
    wd_grid, ws_grid = np.meshgrid(wind_directions, wind_speeds, indexing="ij")
    freq_grid = windrose_interpolant(wd_grid, ws_grid)
    freq_grid = freq_grid / np.sum(freq_grid)  # Normalize to 1.0

    # Calculate farm power baseline
    farm_power_bl = fi_aep_parallel.get_farm_power()
    aep_bl = np.sum(24 * 365 * np.multiply(farm_power_bl, freq_grid))

    # Alternatively to above code, we could calculate AEP using
    # 'fi_aep_parallel.get_farm_AEP(...)' but then we would not have the
    # farm power productions, which we use later on for plotting.

    # First, get baseline AEP, without wake steering
    print(" ")
    print("===========================================================")
    print("Calculating baseline annual energy production (AEP)...")
    print("Baseline AEP: {:.3f} GWh.".format(aep_bl / 1.0e9))
    print("===========================================================")
    print(" ")

    # Load a FLORIS object for yaw optimization
    fi_opt = load_floris()
    wind_directions = np.arange(0.0, 360.0, 3.0)
    wind_speeds = np.arange(6.0, 14.0, 2.0)
    fi_opt.reinitialize(
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        turbulence_intensity=0.08  # Assume 8% turbulence intensity
    )

    # Pour this into a parallel computing interface
    fi_opt_parallel = ParallelComputingInterface(
        fi=fi_opt,
        max_workers=max_workers,
        n_wind_direction_splits=max_workers,
        n_wind_speed_splits=1,
        interface=parallel_interface,
        print_timings=True,
    )

    # Now optimize the yaw angles using the Serial Refine method
    df_opt = fi_opt_parallel.optimize_yaw_angles(
        minimum_yaw_angle=-25.0,
        maximum_yaw_angle=25.0,
        Ny_passes=[5, 4],
        exclude_downstream_turbines=True,
        exploit_layout_symmetry=False,
    )



    # Assume linear ramp up at 5-6 m/s and ramp down at 13-14 m/s,
    # add to table for linear interpolant
    df_copy_lb = df_opt[df_opt["wind_speed"] == 6.0].copy()
    df_copy_ub = df_opt[df_opt["wind_speed"] == 13.0].copy()
    df_copy_lb["wind_speed"] = 5.0
    df_copy_ub["wind_speed"] = 14.0
    df_copy_lb["yaw_angles_opt"] *= 0.0
    df_copy_ub["yaw_angles_opt"] *= 0.0
    df_opt = pd.concat([df_copy_lb, df_opt, df_copy_ub], axis=0).reset_index(drop=True)

    # Deal with 360 deg wrapping: solutions at 0 deg are also solutions at 360 deg
    df_copy_360deg = df_opt[df_opt["wind_direction"] == 0.0].copy()
    df_copy_360deg["wind_direction"] = 360.0
    df_opt = pd.concat([df_opt, df_copy_360deg], axis=0).reset_index(drop=True)

    # Derive linear interpolant from solution space
    yaw_angles_interpolant = LinearNDInterpolator(
        points=df_opt[["wind_direction", "wind_speed"]],
        values=np.vstack(df_opt["yaw_angles_opt"]),
        fill_value=0.0,
    )

    # Get optimized AEP, with wake steering
    yaw_grid = yaw_angles_interpolant(wd_grid, ws_grid)
    farm_power_opt = fi_aep_parallel.get_farm_power(yaw_angles=yaw_grid)
    aep_opt = np.sum(24 * 365 * np.multiply(farm_power_opt, freq_grid))
    aep_uplift = 100.0 * (aep_opt / aep_bl - 1)

    # Alternatively to above code, we could calculate AEP using
    # 'fi_aep_parallel.get_farm_AEP(...)' but then we would not have the
    # farm power productions, which we use later on for plotting.

    print(" ")
    print("===========================================================")
    print("Calculating optimized annual energy production (AEP)...")
    print("Optimized AEP: {:.3f} GWh.".format(aep_opt / 1.0e9))
    print("Relative AEP uplift by wake steering: {:.3f} %.".format(aep_uplift))
    print("===========================================================")
    print(" ")

    # Now calculate helpful variables and then plot wind rose information
    farm_energy_bl = np.multiply(freq_grid, farm_power_bl)
    farm_energy_opt = np.multiply(freq_grid, farm_power_opt)
    df = pd.DataFrame({
        "wd": wd_grid.flatten(),
        "ws": ws_grid.flatten(),
        "freq_val": freq_grid.flatten(),
        "farm_power_baseline": farm_power_bl.flatten(),
        "farm_power_opt": farm_power_opt.flatten(),
        "farm_power_relative": farm_power_opt.flatten() / farm_power_bl.flatten(),
        "farm_energy_baseline": farm_energy_bl.flatten(),
        "farm_energy_opt": farm_energy_opt.flatten(),
        "energy_uplift": (farm_energy_opt - farm_energy_bl).flatten(),
        "rel_energy_uplift": farm_energy_opt.flatten() / np.sum(farm_energy_bl)
    })

    # Plot power and AEP uplift across wind direction
    wd_step = np.diff(fi_aep.floris.flow_field.wind_directions)[0]  # Useful variable for plotting
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
        width=wd_step,
    )
    ax[0].set_ylabel("Power uplift \n at 8 m/s (%)")
    ax[0].grid(True)

    dist = df.groupby("wd").sum().reset_index()
    ax[1].bar(
        x=dist["wd"],
        height=100 * dist["rel_energy_uplift"],
        color="darkgray",
        edgecolor="black",
        width=wd_step,
    )
    ax[1].set_ylabel("Contribution to \n AEP uplift (%)")
    ax[1].grid(True)

    ax[2].bar(
        x=dist["wd"],
        height=dist["freq_val"],
        color="darkgray",
        edgecolor="black",
        width=wd_step,
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
    wd_plot = np.arange(0.0, 360.001, 1.0)
    for ti in range(np.min([fi_aep.floris.farm.n_turbines, 3])):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ws_to_plot = [6.0, 9.0, 12.0]
        colors = ["maroon", "dodgerblue", "grey"]
        styles = ["-o", "-v", "-o"]
        for ii, ws in enumerate(ws_to_plot):
            ax.plot(
                wd_plot,
                yaw_angles_interpolant(wd_plot, ws * np.ones_like(wd_plot))[:, ti],
                styles[ii],
                color=colors[ii],
                markersize=3,
                label="For wind speed of {:.1f} m/s".format(ws),
            )
        ax.set_ylabel("Assigned yaw offsets (deg)")
        ax.set_xlabel("Wind direction (deg)")
        ax.set_title("Turbine {:d}".format(ti))
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

    plt.show()
