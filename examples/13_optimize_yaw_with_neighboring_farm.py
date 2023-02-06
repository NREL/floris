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
from scipy.interpolate import NearestNDInterpolator

from floris.tools import FlorisInterface
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR


"""
This example demonstrates how to perform a yaw optimization and evaluate the performance over a
full wind rose.

The beginning of the file contains the definition of several functions used in the main part of
the script.

Within the main part of the script, we first load the wind rose information.
We then initialize our Floris Interface object. We determine the baseline AEP using the
wind rose information, and then perform the yaw optimization over 72 wind directions with 1
wind speed per direction. The optimal yaw angles are then used to determine yaw angles across
all the wind speeds included in the wind rose. Lastly, the final AEP is calculated and analysis
of the results are shown in several plots.
"""

def load_floris():
    # Load the default example floris object
    fi = FlorisInterface("inputs/gch.yaml") # GCH model matched to the default "legacy_gauss" of V2
    # fi = FlorisInterface("inputs/cc.yaml") # New CumulativeCurl model

    # Specify the full wind farm layout: nominal and neighboring wind farms
    X = np.array(
        [
               0.,   756.,  1512.,  2268.,  3024.,     0.,   756.,  1512.,
            2268.,  3024.,     0.,   756.,  1512.,  2268.,  3024.,     0.,
             756.,  1512.,  2268.,  3024.,  4500.,  5264.,  6028.,  4878.,
               0.,   756.,  1512.,  2268.,  3024.,
        ]
    ) / 1.5
    Y = np.array(
        [
               0.,     0.,     0.,     0.,     0.,   504.,   504.,   504.,
             504.,   504.,  1008.,  1008.,  1008.,  1008.,  1008.,  1512.,
            1512.,  1512.,  1512.,  1512.,  4500.,  4059.,  3618.,  5155.,
            -504.,  -504.,  -504.,  -504.,  -504.,
       ]
    ) / 1.5

    # Turbine weights: we want to only optimize for the first 10 turbines
    turbine_weights = np.zeros(len(X), dtype=int)
    turbine_weights[0:10] = 1.0

    # Now reinitialize FLORIS layout
    fi.reinitialize(layout_x = X, layout_y = Y)

    # And visualize the floris layout
    fig, ax = plt.subplots()
    ax.plot(X[turbine_weights == 0], Y[turbine_weights == 0], 'ro', label="Neighboring farms")
    ax.plot(X[turbine_weights == 1], Y[turbine_weights == 1], 'go', label='Farm subset')
    ax.grid(True)
    ax.set_xlabel("x coordinate (m)")
    ax.set_ylabel("y coordinate (m)")
    ax.legend()

    return fi, turbine_weights


def load_windrose():
    # Load the wind rose information from an external file
    df = pd.read_csv("inputs/wind_rose.csv")
    df = df[(df["ws"] < 22)].reset_index(drop=True)  # Reduce size
    df["freq_val"] = df["freq_val"] / df["freq_val"].sum() # Normalize wind rose frequencies

    # Now put the wind rose information in FLORIS format
    ws_windrose = df["ws"].unique()
    wd_windrose = df["wd"].unique()
    wd_grid, ws_grid = np.meshgrid(wd_windrose, ws_windrose, indexing="ij")

    # Use an interpolant to shape the 'freq_val' vector appropriately. You can
    # also use np.reshape(), but NearestNDInterpolator is more fool-proof.
    freq_interpolant = NearestNDInterpolator(
        df[["ws", "wd"]], df["freq_val"]
    )
    freq = freq_interpolant(wd_grid, ws_grid)
    freq_windrose = freq / freq.sum()  # Normalize to sum to 1.0

    return ws_windrose, wd_windrose, freq_windrose


def optimize_yaw_angles(fi_opt):
    # Specify turbines to optimize
    turbs_to_opt = np.zeros(len(fi_opt.layout_x), dtype=bool)
    turbs_to_opt[0:10] = True

    # Specify turbine weights
    turbine_weights = np.zeros(len(fi_opt.layout_x))
    turbine_weights[turbs_to_opt] = 1.0

    # Specify minimum and maximum allowable yaw angle limits
    minimum_yaw_angle = np.zeros(
        (
            fi_opt.floris.flow_field.n_wind_directions,
            fi_opt.floris.flow_field.n_wind_speeds,
            fi_opt.floris.farm.n_turbines
        )
    )
    maximum_yaw_angle = np.zeros(
        (
            fi_opt.floris.flow_field.n_wind_directions,
            fi_opt.floris.flow_field.n_wind_speeds,
            fi_opt.floris.farm.n_turbines
        )
    )
    maximum_yaw_angle[:, :, turbs_to_opt] = 30.0

    yaw_opt = YawOptimizationSR(
        fi=fi_opt,
        minimum_yaw_angle=minimum_yaw_angle,
        maximum_yaw_angle=maximum_yaw_angle,
        turbine_weights=turbine_weights,
        Ny_passes=[5],
        exclude_downstream_turbines=True,
    )

    df_opt = yaw_opt.optimize()
    yaw_angles_opt = yaw_opt.yaw_angles_opt
    print("Optimization finished.")
    print(" ")
    print(df_opt)
    print(" ")

    # Now create an interpolant from the optimal yaw angles
    def yaw_opt_interpolant(wd, ws):
        # Format the wind directions and wind speeds accordingly
        wd = np.array(wd, dtype=float)
        ws = np.array(ws, dtype=float)

        # Interpolate optimal yaw angles
        x = yaw_opt.fi.floris.flow_field.wind_directions
        nturbs = fi_opt.floris.farm.n_turbines
        y = np.stack(
            [np.interp(wd, x, yaw_angles_opt[:, 0, ti]) for ti in range(nturbs)],
            axis=np.ndim(wd)
        )

        # Now, we want to apply a ramp-up region near cut-in and ramp-down
        # region near cut-out wind speed for the yaw offsets.
        lim = np.ones(np.shape(wd), dtype=float)  # Introduce a multiplication factor

        # Dont do wake steering under 4 m/s or above 14 m/s
        lim[(ws <= 4.0) | (ws >= 14.0)] = 0.0

        # Linear ramp up for the maximum yaw offset between 4.0 and 6.0 m/s
        ids = (ws > 4.0) & (ws < 6.0)
        lim[ids] = (ws[ids] - 4.0) / 2.0

        # Linear ramp down for the maximum yaw offset between 12.0 and 14.0 m/s
        ids = (ws > 12.0) & (ws < 14.0)
        lim[ids] = (ws[ids] - 12.0) / 2.0

        # Copy over multiplication factor to every turbine
        lim = np.expand_dims(lim, axis=np.ndim(wd)).repeat(nturbs, axis=np.ndim(wd))
        lim = lim * 30.0  # These are the limits

        # Finally, Return clipped yaw offsets to the limits
        return np.clip(a=y, a_min=0.0, a_max=lim)

    # Return the yaw interpolant
    return yaw_opt_interpolant


if __name__ == "__main__":
    # Load FLORIS: full farm including neighboring wind farms
    fi, turbine_weights = load_floris()
    nturbs = len(fi.layout_x)

    # Load a dataframe containing the wind rose information
    ws_windrose, wd_windrose, freq_windrose = load_windrose()
    ws_windrose = ws_windrose + 0.001  # Deal with 0.0 m/s discrepancy

    # Create a FLORIS object for AEP calculations
    fi_AEP = fi.copy()
    fi_AEP.reinitialize(wind_speeds=ws_windrose, wind_directions=wd_windrose)

    # And create a separate FLORIS object for optimization
    fi_opt = fi.copy()
    fi_opt.reinitialize(
        wind_directions=np.arange(0.0, 360.0, 3.0),
        wind_speeds=[8.0]
    )

    # First, get baseline AEP, without wake steering
    print(" ")
    print("===========================================================")
    print("Calculating baseline annual energy production (AEP)...")
    aep_bl_subset = 1.0e-9 * fi_AEP.get_farm_AEP(
        freq=freq_windrose,
        turbine_weights=turbine_weights
    )
    print("Baseline AEP for subset farm: {:.3f} GWh.".format(aep_bl_subset))
    print("===========================================================")
    print(" ")

    # Now optimize the yaw angles using the Serial Refine method. We first
    # create a copy of the floris object for optimization purposes and assign
    # it the atmospheric conditions for which we want to optimize. Typically,
    # the optimal yaw angles are very insensitive to the actual wind speed,
    # and hence we only optimize for a single wind speed of 8.0 m/s. We assume
    # that the optimal yaw angles at 8.0 m/s are also optimal at other wind
    # speeds between 4 and 12 m/s.
    print("Now starting yaw optimization for the entire wind rose for farm subset...")

    # In this hypothetical case, we can only control the yaw angles of the
    # turbines of the wind farm subset (i.e., the first 10 wind turbines).
    # Hence, we constrain the yaw angles of the neighboring wind farms to 0.0.
    turbs_to_opt = (turbine_weights > 0.0001)

    # Optimize yaw angles while including neighboring farm
    yaw_opt_interpolant = optimize_yaw_angles(fi_opt=fi_opt)

    # Optimize yaw angles while ignoring neighboring farm
    fi_opt_subset = fi_opt.copy()
    fi_opt_subset.reinitialize(
        layout_x = fi.layout_x[turbs_to_opt],
        layout_y = fi.layout_y[turbs_to_opt]
    )
    yaw_opt_interpolant_nonb = optimize_yaw_angles(fi_opt=fi_opt_subset)

    # Use interpolant to get optimal yaw angles for fi_AEP object
    X, Y = np.meshgrid(
        fi_AEP.floris.flow_field.wind_directions,
        fi_AEP.floris.flow_field.wind_speeds,
        indexing="ij"
    )
    yaw_angles_opt_AEP = yaw_opt_interpolant(X, Y)
    yaw_angles_opt_nonb_AEP = np.zeros_like(yaw_angles_opt_AEP)  # nonb = no neighbor
    yaw_angles_opt_nonb_AEP[:, :, turbs_to_opt] = yaw_opt_interpolant_nonb(X, Y)

    # Now get AEP with optimized yaw angles
    print(" ")
    print("===========================================================")
    print("Calculating annual energy production with wake steering (AEP)...")
    aep_opt_subset_nonb = 1.0e-9 * fi_AEP.get_farm_AEP(
        freq=freq_windrose,
        turbine_weights=turbine_weights,
        yaw_angles=yaw_angles_opt_nonb_AEP,
    )
    aep_opt_subset = 1.0e-9 * fi_AEP.get_farm_AEP(
        freq=freq_windrose,
        turbine_weights=turbine_weights,
        yaw_angles=yaw_angles_opt_AEP,
    )
    uplift_subset_nonb = 100.0 * (aep_opt_subset_nonb - aep_bl_subset) / aep_bl_subset
    uplift_subset = 100.0 * (aep_opt_subset - aep_bl_subset) / aep_bl_subset
    print(
        "Optimized AEP for subset farm (including neighbor farms' wakes): "
        f"{aep_opt_subset_nonb:.3f} GWh (+{uplift_subset_nonb:.2f}%)."
    )
    print(
        "Optimized AEP for subset farm (ignoring neighbor farms' wakes): "
        f"{aep_opt_subset:.3f} GWh (+{uplift_subset:.2f}%)."
    )
    print("===========================================================")
    print(" ")

    # Plot power and AEP uplift across wind direction at wind_speed of 8 m/s
    X, Y = np.meshgrid(
        fi_opt.floris.flow_field.wind_directions,
        fi_opt.floris.flow_field.wind_speeds,
        indexing="ij",
    )
    yaw_angles_opt = yaw_opt_interpolant(X, Y)

    yaw_angles_opt_nonb = np.zeros_like(yaw_angles_opt)  # nonb = no neighbor
    yaw_angles_opt_nonb[:, :, turbs_to_opt] = yaw_opt_interpolant_nonb(X, Y)

    fi_opt = fi_opt.copy()
    fi_opt.calculate_wake(yaw_angles=np.zeros_like(yaw_angles_opt))
    farm_power_bl_subset = fi_opt.get_farm_power(turbine_weights).flatten()

    fi_opt = fi_opt.copy()
    fi_opt.calculate_wake(yaw_angles=yaw_angles_opt)
    farm_power_opt_subset = fi_opt.get_farm_power(turbine_weights).flatten()

    fi_opt = fi_opt.copy()
    fi_opt.calculate_wake(yaw_angles=yaw_angles_opt_nonb)
    farm_power_opt_subset_nonb = fi_opt.get_farm_power(turbine_weights).flatten()

    fig, ax = plt.subplots()
    ax.bar(
        x=fi_opt.floris.flow_field.wind_directions - 0.65,
        height=100.0 * (farm_power_opt_subset / farm_power_bl_subset - 1.0),
        edgecolor="black",
        width=1.3,
        label="Including wake effects of neighboring farms"
    )
    ax.bar(
        x=fi_opt.floris.flow_field.wind_directions + 0.65,
        height=100.0 * (farm_power_opt_subset_nonb / farm_power_bl_subset - 1.0),
        edgecolor="black",
        width=1.3,
        label="Ignoring neighboring farms"
    )
    ax.set_ylabel("Power uplift \n at 8 m/s (%)")
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("Wind direction (deg)")

    plt.show()
