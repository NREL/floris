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


# NOTE: This example requires an additional module, `streamlit`.
#
# TO USE THIS FROM COMMAND TYPE:
#    streamlit run compare_models.py

import time

import numpy as np
import matplotlib.pyplot as plt

import streamlit as st
import floris.tools as wfct


# Fixed parameters
minSpeed = 4
maxSpeed = 8.0
D = 126.0
TI = 0.06

# User inputs
D = 126.0
nturbs_x = 5
nturbs_y = 1
x_spacing = 5 * D
y_spacing = 3 * D

# Generate layout
layout_x = [i * x_spacing for j in range(nturbs_y) for i in range(nturbs_x)]
layout_y = [
    j * y_spacing + i * 0.0 * D for j in range(nturbs_y) for i in range(nturbs_x)
]
layout_array = [layout_x, layout_y]

# Overall options
ws = st.sidebar.slider("Wind Speed", 5.0, 15.0, 8.0, step=0.5)
wd = st.sidebar.slider("Wind Direction", 260.0, 280.0, 270.0, step=0.5)

# Options For 1 turbine case
yaw = st.sidebar.slider("Front turbine yaw angle", -30.0, 30.0, 20.0, step=1.0)
x_loc = st.sidebar.slider("D downstream from last turbine", 2.0, 30.0, 7.0, step=1.0)
spacing = st.sidebar.slider("Inter-Turbine Spacing", 2.0, 12.0, 7.0, step=1.0)
n_row = st.sidebar.slider("Num Rows in Deep Array", 2, 12, 5, step=1)
n_col = st.sidebar.slider("Num Columns in Deep Array", 1, 12, 1, step=1)
resolution = st.sidebar.slider("Resolution", 25, 200, 50, step=1)

# Initialize the FLORIS models
# fi_jensen = wfct.floris_interface.FlorisInterface("floris_models/jensen.json")
# fi_turbopark = wfct.floris_interface.FlorisInterface("floris_models/turbopark.json")
fi_gch = wfct.floris_interface.FlorisInterface("floris_models/gch.json")
fi_gch.reinitialize_flow_field(
    wind_speed=ws, wind_direction=wd, turbulence_intensity=TI, layout_array=layout_array
)

fi_gauss = wfct.floris_interface.FlorisInterface("floris_models/gauss_legacy.json")
# fi_gauss.set_gch(enable=False)
fi_gauss.reinitialize_flow_field(
    wind_speed=ws, wind_direction=wd, turbulence_intensity=TI, layout_array=layout_array
)

fi_gauss_linear = wfct.floris_interface.FlorisInterface("floris_models/gch_linear.json")
fi_gauss_linear.set_gch(enable=False)
fi_gauss_linear.reinitialize_flow_field(
    wind_speed=ws, wind_direction=wd, turbulence_intensity=TI, layout_array=layout_array
)

fi_gauss_SS = wfct.floris_interface.FlorisInterface("floris_models/gch.json")
fi_gauss_SS.set_gch(enable=False)
fi_gauss_SS.set_gch_secondary_steering(enable=True)
fi_gauss_SS.reinitialize_flow_field(
    wind_speed=ws, wind_direction=wd, turbulence_intensity=TI, layout_array=layout_array
)

fi_gauss_YAR = wfct.floris_interface.FlorisInterface("floris_models/gch.json")
fi_gauss_YAR.set_gch(enable=False)
fi_gauss_YAR.set_gch_yaw_added_recovery(enable=True)
fi_gauss_YAR.reinitialize_flow_field(
    wind_speed=ws, wind_direction=wd, turbulence_intensity=TI, layout_array=layout_array
)

fi_gch.reinitialize_flow_field(
    wind_speed=ws, wind_direction=wd, turbulence_intensity=TI, layout_array=layout_array
)

# fi_jensen_linear = wfct.floris_interface.FlorisInterface("floris_models/jensen_linear.json")
fi_gch_linear = wfct.floris_interface.FlorisInterface("floris_models/gch_linear.json")
fi_gch_linear.reinitialize_flow_field(
    wind_speed=ws, wind_direction=wd, turbulence_intensity=TI, layout_array=layout_array
)


fi_gauss_cumulative_alpha1 = wfct.floris_interface.FlorisInterface(
    "floris_models/gch.json"
)
fi_gauss_cumulative_alpha1.floris.farm.set_wake_model("gauss_cumulative")
fi_gc_params = {
    "Wake Turbulence Parameters": {
        "ti_ai": 0.83,
        "ti_constant": 0.66,
        "ti_downstream": -0.32,
        "ti_initial": 0.03,
    },
    "Wake Velocity Parameters": {"sigma_gch": True},
}
fi_gauss_cumulative_alpha1.set_model_parameters(fi_gc_params)
fi_gauss_cumulative_alpha1.set_gch(enable=False)
# fi_gauss_cumulative_alpha1.floris.farm.flow_field.solver = "gauss_cumulative"
fi_gauss_cumulative_alpha1.floris.farm.wake.solver = "cumulative"
fi_gauss_cumulative_alpha1.reinitialize_flow_field(
    wind_speed=ws, wind_direction=wd, turbulence_intensity=TI, layout_array=layout_array
)


fi_gauss_cumulative_alpha2 = wfct.floris_interface.FlorisInterface(
    "floris_models/gch.json"
)
fi_gauss_cumulative_alpha2.floris.farm.set_wake_model("gauss_cumulative")
fi_gc_params = {
    "Wake Turbulence Parameters": {
        "ti_ai": 0.83,
        "ti_constant": 0.66,
        "ti_downstream": -0.32,
        "ti_initial": 0.03,
    },
    "Wake Velocity Parameters": {"alpha_mod": 2.0},
}
fi_gauss_cumulative_alpha2.set_model_parameters(fi_gc_params)
fi_gauss_cumulative_alpha2.set_gch(enable=False)
# fi_gauss_cumulative_alpha2.floris.farm.flow_field.solver = "gauss_cumulative"
fi_gauss_cumulative_alpha2.floris.farm.wake.solver = "cumulative"
fi_gauss_cumulative_alpha2.reinitialize_flow_field(
    wind_speed=ws, wind_direction=wd, turbulence_intensity=TI, layout_array=layout_array
)


fi_gauss_cumulative_alpha1_SS = wfct.floris_interface.FlorisInterface(
    "floris_models/gch.json"
)
fi_gauss_cumulative_alpha1_SS.floris.farm.set_wake_model("gauss_cumulative")
fi_gc_params = {
    "Wake Turbulence Parameters": {
        "ti_ai": 0.83,
        "ti_constant": 0.66,
        "ti_downstream": -0.32,
        "ti_initial": 0.03,
    },
    "Wake Velocity Parameters": {"sigma_gch": True},
}
fi_gauss_cumulative_alpha1_SS.set_model_parameters(fi_gc_params)
# fi_gauss_cumulative_alpha1_SS.set_gch(enable=False)
fi_gauss_cumulative_alpha1_SS.set_gch_yaw_added_recovery(enable=False)
fi_gauss_cumulative_alpha1_SS.set_gch_secondary_steering(enable=True)
# fi_gauss_cumulative_alpha1_FS.floris.farm.flow_field.solver = "gauss_cumulative"
fi_gauss_cumulative_alpha1_SS.floris.farm.wake.solver = "cumulative"
fi_gauss_cumulative_alpha1_SS.reinitialize_flow_field(
    wind_speed=ws, wind_direction=wd, turbulence_intensity=TI, layout_array=layout_array
)

fi_gauss_cumulative_alpha1_YAR = wfct.floris_interface.FlorisInterface(
    "floris_models/gch.json"
)
fi_gauss_cumulative_alpha1_YAR.floris.farm.set_wake_model("gauss_cumulative")
fi_gc_params = {
    "Wake Turbulence Parameters": {
        "ti_ai": 0.83,
        "ti_constant": 0.66,
        "ti_downstream": -0.32,
        "ti_initial": 0.03,
    },
    "Wake Velocity Parameters": {"sigma_gch": True},
}
fi_gauss_cumulative_alpha1_YAR.set_model_parameters(fi_gc_params)
# fi_gauss_cumulative_alpha1_YAR.set_gch(enable=False)
fi_gauss_cumulative_alpha1_YAR.set_gch_yaw_added_recovery(enable=True)
fi_gauss_cumulative_alpha1_YAR.set_gch_secondary_steering(enable=False)
# fi_gauss_cumulative_alpha1_FS.floris.farm.flow_field.solver = "gauss_cumulative"
fi_gauss_cumulative_alpha1_YAR.floris.farm.wake.solver = "cumulative"
fi_gauss_cumulative_alpha1_YAR.reinitialize_flow_field(
    wind_speed=ws, wind_direction=wd, turbulence_intensity=TI, layout_array=layout_array
)

fi_gauss_cumulative_alpha1_GCH = wfct.floris_interface.FlorisInterface(
    "floris_models/gch.json"
)
fi_gauss_cumulative_alpha1_GCH.floris.farm.set_wake_model("gauss_cumulative")
fi_gc_params = {
    "Wake Turbulence Parameters": {
        "ti_ai": 0.83,
        "ti_constant": 0.66,
        "ti_downstream": -0.32,
        "ti_initial": 0.03,
    },
    "Wake Velocity Parameters": {"sigma_gch": True},
}
fi_gauss_cumulative_alpha1_GCH.set_model_parameters(fi_gc_params)
fi_gauss_cumulative_alpha1_GCH.set_gch(enable=True)
# fi_gauss_cumulative_alpha1_GCH.set_gch_yaw_added_recovery(enable=True)
# fi_gauss_cumulative_alpha1_GCH.set_gch_secondary_steering(enable=True)
# fi_gauss_cumulative_alpha1_FS.floris.farm.flow_field.solver = "gauss_cumulative"
fi_gauss_cumulative_alpha1_GCH.floris.farm.wake.solver = "cumulative"
fi_gauss_cumulative_alpha1_GCH.reinitialize_flow_field(
    wind_speed=ws, wind_direction=wd, turbulence_intensity=TI, layout_array=layout_array
)

# fi_gauss_cumulative_alpha2_FS = wfct.floris_interface.FlorisInterface(
#     "floris_models/gch.json"
# )
# fi_gauss_cumulative_alpha2_FS.floris.farm.set_wake_model("gauss_cumulative")
# fi_gc_params = {
#     "Wake Turbulence Parameters": {
#         "ti_ai": 0.83,
#         "ti_constant": 0.66,
#         "ti_downstream": -0.32,
#         "ti_initial": 0.03,
#     },
#     "Wake Velocity Parameters": {
#         "alpha_mod": 2.0,
#         "sigma_gch": True,
#     },
# }
# fi_gauss_cumulative_alpha2_FS.set_model_parameters(fi_gc_params)
# fi_gauss_cumulative_alpha2_FS.set_gch(enable=False)
# fi_gauss_cumulative_alpha2_FS.set_gch_yaw_added_recovery(enable=False)
# fi_gauss_cumulative_alpha2_FS.set_gch_secondary_steering(enable=True)
# # fi_gauss_cumulative_alpha2_FS.floris.farm.flow_field.solver = "gauss_cumulative"
# fi_gauss_cumulative_alpha2_FS.floris.farm.wake.solver = "cumulative"
# fi_gauss_cumulative_alpha2_FS.reinitialize_flow_field(
#     wind_speed=ws, wind_direction=wd, turbulence_intensity=TI, layout_array=layout_array
# )


fi_turbopark = wfct.floris_interface.FlorisInterface("floris_models/turbopark.json")


# fi_gauss_legacy.floris.farm.set_wake_model("gauss_legacy")
# # fi_gauss_legacy.floris.farm.flow_field.solver = "floris"
# fi_gauss_legacy.floris.farm.wake.solver = "sequential"
# fi_gauss_legacy.reinitialize_flow_field(
#     wind_speed=ws, wind_direction=wd, turbulence_intensity=TI, layout_array=layout_array
# )


# Set up some lists
# fi_list = [fi_jensen, fi_turbopark, fi_gch,fi_gch_linear, fi_gauss_cumulative]
# label_list = ["Jensen",  "TurbOPark", "GCH", "GCH (Linear)", "Cumulative"]
# color_list = ["b", "r", "g", "g","m"]
# ls_list = ["-","-","-","--","-"]

# fi_list = [
#     fi_gch,
#     fi_gch_linear,
#     fi_gauss_cumulative_alpha1,
#     fi_gauss_cumulative_alpha1_SS,
#     fi_gauss_cumulative_alpha1_YAR,
#     fi_gauss_cumulative_alpha1_GCH,
#     fi_turbopark,
# ]
# label_list = [
#     "GCH",
#     "GCH (Linear)",
#     "Cumulative: Alpha 1",
#     "Cumulative: Alpha 1, SS",
#     "Cumulative: Alpha 1, YAR",
#     "Cumulative: Alpha 1, GCH",
#     "TurbOPark",
# ]
# color_list = ["g", "g", "m", "y", "k", "c", "b"]
# ls_list = ["-", "--", "--", "-.", "--", "-.", "--"]

# fi_list = [
#     fi_gauss,
#     fi_gauss_SS,
#     fi_gauss_YAR,
#     fi_gch,
#     fi_gch_linear,
#     fi_gauss_cumulative_alpha1,
#     fi_gauss_cumulative_alpha1_SS,
#     fi_gauss_cumulative_alpha1_YAR,
#     fi_gauss_cumulative_alpha1_GCH,
#     fi_turbopark,
# ]
# label_list = [
#     "Gauss",
#     "Gauss_SS",
#     "Gauss_YAR",
#     "GCH",
#     "GCH (Linear)",
#     "Cumulative: Alpha 1",
#     "Cumulative: Alpha 1, SS",
#     "Cumulative: Alpha 1, YAR",
#     "Cumulative: Alpha 1, GCH",
#     "TurbOPark",
# ]
# color_list = ["r", "b", "orange", "g", "g", "m", "y", "k", "c", "b"]
# ls_list = ["-", "--", "-.", "-", "--", "--", "-.", "--", "-.", "--"]

# fi_list = [
#     fi_gch,
#     fi_gch_linear,
#     fi_gauss_cumulative_alpha1,
#     fi_gauss_cumulative_alpha1_GCH,
#     fi_turbopark,
# ]
# label_list = [
#     "GCH (Sum of Squares)",
#     "GCH (Linear)",
#     "Cumulative (Alpha = 1)",
#     "Cumulative (Alpha = 1, GCH)",
#     "TurbOPark",
# ]
# color_list = ["g", "g", "m", "m", "b"]
# ls_list = ["-", "--", "-", "--", "-"]

fi_list = [
    fi_gauss,
    fi_gch,
    fi_gauss_cumulative_alpha1,
    fi_gauss_cumulative_alpha1_SS,
]
label_list = [
    "Gauss",
    "GCH",
    "Cumulative: Alpha 1",
    "Cumulative: Alpha 1, SS",
]
color_list = ["g", "m", "k", "k"]
ls_list = ["-", "-", "-", "--"]

# Other parameters
num_models = len(fi_list)


# SINGLE WAKE CASE

# st.write("# Single Wake")

# # Determine sweep location
# sweep_loc = x_loc * D

# # Calibrate models to single wake
# fig_cut, axarr_cut = plt.subplots(1, num_models)
# fig_rat, ax_rat = plt.subplots(1, 1)
# for i in range(num_models):

#     # Grab the model info
#     fi = fi_list[i]
#     label = label_list[i]
#     color = color_list[i]
#     ls = ls_list[i]

#     # Configure model
#     fi.reinitialize_flow_field(wind_speed=ws,wind_direction=wd, layout_array=[[0], [0]])
#     fi.calculate_wake(yaw_angles=[yaw])

#     # Show the hor plane
#     ax = axarr_cut[i]
#     hor_plane = fi.get_hor_plane()
#     wfct.visualization.visualize_cut_plane(
#         hor_plane, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
#     )
#     ax.set_title(label)
#     ax.axvline(sweep_loc, color="w", ls="--")

#     # Perform the sweep (cross wind)
#     sweep_locations = np.linspace(-3, 3.25, resolution)
#     power_out = np.zeros_like(sweep_locations)

#     for y_idx, y_loc in enumerate(sweep_locations):

#         fi.reinitialize_flow_field(layout_array=([0, sweep_loc], [0, y_loc * D],))
#         fi.calculate_wake(yaw_angles=[yaw, 0])
#         power_out[y_idx] = fi.get_turbine_power()[1] / fi.get_turbine_power()[0]

#     ax_rat.plot(sweep_locations, power_out, color=color, label=label, ls=ls)
#     ax_rat.legend()
#     ax_rat.grid(True)
#     ax_rat.set_title("Normalized Power at %.1f D downstream" % x_loc)
#     ax_rat.set_ylabel("Normalized Power (-)")
#     ax_rat.set_xlabel("Cross-Stream Location (D)")

# st.write(fig_cut)
# st.write(fig_rat)


# ########### SINGLE WAKE CASE LONG

# st.write("# Single Wake Long wise")

# # Determine sweep location
# y_loc_1 = 0
# sweep_loc = y_loc_1 * D

# # Calibrate models to single wake
# fig_rat, ax_rat = plt.subplots(1, 1)
# for i in range(num_models):

#     # Grab the model info
#     fi = fi_list[i]
#     label = label_list[i]
#     color = color_list[i]

#     # Configure model
#     fi.reinitialize_flow_field(wind_speed=ws,wind_direction=wd, layout_array=[[0], [0]])
#     fi.calculate_wake(yaw_angles=[yaw_1])

#     # Perform the sweep (cross wind)
#     sweep_locations = np.arange(2, 40, 0.5)
#     power_out = np.zeros_like(sweep_locations)

#     for x_idx, x_loc in enumerate(sweep_locations):

#         fi.reinitialize_flow_field(layout_array=[[0, x_loc * D], [0, 0]])

#         fi.calculate_wake(yaw_angles=[0, 0])

#         power_out[x_idx] = fi.get_turbine_power()[1] / fi.get_turbine_power()[0]

#     ax_rat.plot(sweep_locations, power_out, color=color, label=label)
#     ax_rat.legend()
#     ax_rat.grid(True)
#     ax_rat.set_title("Normalized Power at %.1f D downstream" % x_loc_1)
#     ax_rat.set_ylabel("Normalized Power (-)")
#     ax_rat.set_xlabel("Cross-Stream Location (D)")


# st.write(fig_rat)


# Double WAKE CASE

# st.write("# Double Wake")


# # Determine locations
# t1_x_loc = spacing * D
# sweep_loc = t1_x_loc + x_loc * D

# # Calibrate models to single wake
# fig_cut, axarr_cut = plt.subplots(1, num_models)
# fig_rat, ax_rat = plt.subplots(1, 1)
# for i in range(num_models):

#     # Grab the model info
#     fi = fi_list[i]
#     label = label_list[i]
#     color = color_list[i]
#     ls = ls_list[i]

#     # Configure model
#     fi.reinitialize_flow_field(
#         wind_speed=ws,wind_direction=wd, layout_array=[[0, t1_x_loc], [0, 0]]
#     )
#     fi.calculate_wake(yaw_angles=[yaw, 0])

#     # Show the hor plane
#     ax = axarr_cut[i]
#     hor_plane = fi.get_hor_plane()
#     wfct.visualization.visualize_cut_plane(
#         hor_plane, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
#     )
#     ax.set_title(label)
#     ax.axvline(sweep_loc, color="w", ls="--")
#     ax.axhline(0, color="w", ls="--")

#     # Perform the sweep

#     sweep_locations = np.linspace(-3, 3.25, resolution)
#     power_out = np.zeros_like(sweep_locations)

#     for y_idx, y_loc in enumerate(sweep_locations):

#         fi.reinitialize_flow_field(
#             layout_array=([0, t1_x_loc, sweep_loc], [0, 0, y_loc * D],)
#         )
#         fi.calculate_wake(yaw_angles=[yaw, 0, 0])
#         power_out[y_idx] = fi.get_turbine_power()[2] / fi.get_turbine_power()[0]

#     ax_rat.plot(sweep_locations, power_out, color=color, label=label, ls=ls)
#     ax_rat.legend()
#     ax_rat.grid(True)
#     ax_rat.set_title("Normalized Power at %.1f D downstream" % x_loc)
#     ax_rat.set_ylabel("Normalized Power (-)")
#     ax_rat.set_xlabel("Cross-Stream Location (D)")

# st.write(fig_cut)
# st.write(fig_rat)


# Deep Array CASE

st.write("# Deep Array Wake")

# Calculate turbines
deep_x = []
deep_y = []
for i in range(n_row):
    for j in range(n_col):
        deep_x.append(i * D * spacing)
        deep_y.append(j * D * spacing + 0)

deep_x = np.array(deep_x)
deep_y = np.array(deep_y)
deep_x = np.array(layout_x)
deep_y = np.array(layout_y)
# yaw_array = np.zeros_like(deep_x)
# yaw_array[deep_x == np.min(deep_x)] = yaw

yaw_array = np.array([20.0, 20.0, 17.0, 12.0, 0.0])
# yaw_array = np.array([20.0])

# Determine locations
sweep_loc = np.max(deep_x) + x_loc * D
sweep_center = np.mean(np.array(deep_y))

# Calibrate models to single wake
fig_cut, axarr_cut = plt.subplots(1, num_models, figsize=(15, 3))
fig_rat, ax_rat = plt.subplots(1, 1)
time_array = []
for i in range(num_models):

    # Grab the model info
    fi = fi_list[i]
    label = label_list[i]
    color = color_list[i]
    ls = ls_list[i]

    # Configure model
    fi.reinitialize_flow_field(
        wind_speed=ws, wind_direction=wd, layout_array=[deep_x, deep_y]
    )
    time_start = time.perf_counter()
    fi.calculate_wake(yaw_angles=yaw_array)  # np.zeros_like(deep_x))
    time_end = time.perf_counter()
    time_array.append(time_end - time_start)

    # Show the hor plane
    ax = axarr_cut[i]
    hor_plane = fi.get_hor_plane()
    wfct.visualization.visualize_cut_plane(
        hor_plane, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
    )
    ax.set_title(label)
    ax.axvline(sweep_loc, color="w", ls="--")
    ax.axhline(sweep_center, color="w", ls="--")

    # Perform the sweep

    # sweep_locations = np.arange(np.min(deep_y) - 2, np.max(deep_y) + 2.5, 0.25)
    sweep_locations = np.linspace(
        np.min(deep_y) / D - 2, np.max(deep_y) / D + 2.5, resolution
    )
    power_out = np.zeros_like(sweep_locations)

    x_locs = [xx for xx in deep_x] + [sweep_loc]
    y_locs = [xx for xx in deep_y] + [0]
    yaw_array = [xx for xx in yaw_array] + [0]

    for y_idx, y_loc in enumerate(sweep_locations):

        y_locs[-1] = y_loc * D
        fi.reinitialize_flow_field(layout_array=(x_locs, y_locs))
        fi.calculate_wake(yaw_angles=yaw_array)
        power_out[y_idx] = fi.get_turbine_power()[-1] / fi.get_turbine_power()[0]

    ax_rat.plot(sweep_locations, power_out, color=color, label=label, ls=ls)
    ax_rat.legend()
    ax_rat.grid(True)
    ax_rat.set_title("Normalized Power at %.1f D downstream" % x_loc)
    ax_rat.set_ylabel("Normalized Power (-)")
    ax_rat.set_xlabel("Cross-Stream Location (D)")

st.write(fig_cut)
st.write(fig_rat)

print(label_list)
print(time_array)

print("gauss turbs: ", fi_gauss.get_turbine_power())
print("gauss_SS turbs: ", fi_gauss_SS.get_turbine_power())
print("gauss_YAR turbs: ", fi_gauss_YAR.get_turbine_power())
print("GCH turbs: ", fi_gch.get_turbine_power())

plt.show()
