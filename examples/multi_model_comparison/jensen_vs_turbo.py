# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


# NOTE: This example requires an additional module, `streamlit`.
#
# TO USE THIS FROM COMMAND TYPE:
#    streamlit run interactive_visual.py

import numpy as np

import floris.tools as wfct
import matplotlib.pyplot as plt


# Fixed parameters
minSpeed = 4
maxSpeed = 8.0
D = 126.0
ws = 8.0
yaw_1 = 0.0


# Initialize the FLORIS models
fi_jensen = wfct.floris_interface.FlorisInterface("floris_models/jensen.json")
fi_turbopark = wfct.floris_interface.FlorisInterface("floris_models/turbopark.json")
# fi_gch = wfct.floris_interface.FlorisInterface("floris_models/gch.json")

# # Set up some lists
# fi_list = [fi_jensen, fi_turbopark, fi_gch]
# label_list = ['Jensen','TurbOPark','GCH']
# color_list = ['b','r','g']

fi_list = [fi_jensen, fi_turbopark]
label_list = ["Jensen", "TurbOPark"]
color_list = ["b", "r"]

# Other parameters
num_models = len(fi_list)


########### SINGLE WAKE CASE

# Determine sweep location
x_loc_1 = 7
sweep_loc = x_loc_1 * D

# Calibrate models to single wake
fig_cut, axarr_cut = plt.subplots(2, 1)
fig_rat, ax_rat = plt.subplots(1, 1)
fig_bound, ax_bound = plt.subplots(1, 1)

for i in range(num_models):

    # Grab the model info
    fi = fi_list[i]
    label = label_list[i]
    color = color_list[i]

    # Configure model
    fi.reinitialize_flow_field(
        # wind_speed=ws, layout_array=[[0], [0]]
        wind_speed=ws,
        layout_array=[[0, D * 60, D * 60, D * 60], [0, 0, 5 * D, -5 * D]],
    )
    fi.calculate_wake(yaw_angles=[yaw_1])

    # Show the hor plane
    ax = axarr_cut[i]
    hor_plane = fi.get_hor_plane()
    wfct.visualization.visualize_cut_plane(
        hor_plane, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
    )
    ax.set_title(label)
    ax.axvline(sweep_loc, color="w", ls="--")

    # Show the boundaries
    ax = ax_bound
    df = hor_plane.df.copy()
    df = df[df.u < df.u.max()]
    df = df.groupby("x1").max().reset_index()
    ax.plot(df.x1 / D, df.x2 / D, color=color, label=label)

    df = hor_plane.df.copy()
    df = df[df.u < df.u.max()]
    df = df.groupby("x1").min().reset_index()
    ax.plot(df.x1 / D, df.x2 / D, color=color, label="_nolegend_")
    ax.legend()
    ax.grid(True)

    # Perform the sweep (cross wind)
    sweep_locations = np.arange(-3, 3.25, 0.25)
    power_out = np.zeros_like(sweep_locations)

    for y_idx, y_loc in enumerate(sweep_locations):

        fi.reinitialize_flow_field(layout_array=([0, sweep_loc], [0, y_loc * D],))
        fi.calculate_wake(yaw_angles=[yaw_1, 0])
        power_out[y_idx] = fi.get_turbine_power()[1] / fi.get_turbine_power()[0]

    ax_rat.plot(sweep_locations, power_out, color=color, label=label)
    ax_rat.legend()
    ax_rat.grid(True)
    ax_rat.set_title("Normalized Power at %.1f D downstream" % x_loc_1)
    ax_rat.set_ylabel("Normalized Power (-)")
    ax_rat.set_xlabel("Cross-Stream Location (D)")


########### SINGLE WAKE CASE LONG

# Determine sweep location
y_loc_1 = 0
sweep_loc = y_loc_1 * D

# Calibrate models to single wake
fig_rat, ax_rat = plt.subplots(1, 1)
for i in range(num_models):

    # Grab the model info
    fi = fi_list[i]
    label = label_list[i]
    color = color_list[i]

    # Configure model
    fi.reinitialize_flow_field(wind_speed=ws, layout_array=[[0], [0]])
    fi.calculate_wake(yaw_angles=[yaw_1])

    # Perform the sweep (cross wind)
    sweep_locations = np.arange(1, 40, 0.5)
    power_out = np.zeros_like(sweep_locations)

    for x_idx, x_loc in enumerate(sweep_locations):

        fi.reinitialize_flow_field(layout_array=[[0, x_loc * D], [0, 0]])

        fi.calculate_wake(yaw_angles=[0, 0])

        power_out[x_idx] = fi.get_turbine_power()[1] / fi.get_turbine_power()[0]

    ax_rat.plot(sweep_locations, power_out, color=color, label=label)
    ax_rat.legend()
    ax_rat.grid(True)
    # ax_rat.set_title("Normalized Power at %.1f D downstream" % x_loc_1)
    ax_rat.set_ylabel("Normalized Power (-)")
    ax_rat.set_xlabel("Downstream Location (D)")


plt.show()
