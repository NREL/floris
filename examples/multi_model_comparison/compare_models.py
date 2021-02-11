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

import matplotlib.pyplot as plt

import streamlit as st
import floris.tools as wfct
import numpy as np



# Fixed parameters
minSpeed = 4
maxSpeed = 8.0
D = 126.

# Overall options
ws = st.sidebar.slider("Wind Speed", 5., 15., 8.0, step=0.5)

# Options For 1 turbine case
yaw_1 = st.sidebar.slider("1T: Yaw angle", -30., 30., 0., step=1.)
x_loc_1 = st.sidebar.slider("1T: D downstream", 2., 30., 7.0, step=1.0)

# Options for 2 turbine case
x_loc_2 = st.sidebar.slider("2T: D downstream", 2., 30., 7.0, step=1.0)
spacing_2 = st.sidebar.slider("2T: D Spacing", 2., 12., 7.0, step=1.0)
offset_2 = st.sidebar.slider("2T: D Offset", -2., 2., 0.0, step=0.25)

# Options for deep array case
x_loc_3 = st.sidebar.slider("Deep: D downstream", 2., 50., 7.0, step=1.0)
n_row = st.sidebar.slider("Deep: Num Rows", 2, 12, 2, step=1)
spacing_3 = st.sidebar.slider("Deep: D Spacing", 2., 12., 7.0, step=1.0)
offset_3 = st.sidebar.slider("Deep: D Offset", -2., 2., 0.0, step=0.25)

# Initialize the FLORIS models
fi_jensen = wfct.floris_interface.FlorisInterface("floris_models/jensen.json")
fi_turbopark = wfct.floris_interface.FlorisInterface("floris_models/turbopark.json")
fi_gch = wfct.floris_interface.FlorisInterface("floris_models/gch.json")

# Set up some lists
fi_list = [fi_jensen, fi_turbopark, fi_gch]
label_list = ['Jensen','TurbOPark','GCH']
color_list = ['b','r','g']

# Other parameters
num_models = len(fi_list)


########### SINGLE WAKE CASE

st.write("# Single Wake")

# Determine sweep location
sweep_loc = x_loc_1 * D

# Calibrate models to single wake
fig_cut, axarr_cut = plt.subplots(1,3)
fig_rat, ax_rat = plt.subplots(1,1)
for i in range(num_models):

    # Grab the model info
    fi = fi_list[i]
    label = label_list[i]
    color = color_list[i]

    # Configure model
    fi.reinitialize_flow_field(
        wind_speed=ws, layout_array=[[0], [0]]
    )
    fi.calculate_wake(yaw_angles=[yaw_1])

    # Show the hor plane
    ax = axarr_cut[i]
    hor_plane = fi.get_hor_plane()
    wfct.visualization.visualize_cut_plane(
        hor_plane, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
    )
    ax.set_title(label)
    ax.axvline(sweep_loc,color='w',ls='--')

    # Perform the sweep (cross wind)
    sweep_locations = np.arange(-3, 3.25, 0.25)
    power_out = np.zeros_like(sweep_locations)

    for y_idx, y_loc in enumerate(sweep_locations):

        fi.reinitialize_flow_field(
            layout_array=(
                [
                    0,
                    sweep_loc
                ],
                [0, y_loc * D],
            )
        )
        fi.calculate_wake(yaw_angles=[yaw_1,0])
        power_out[y_idx] = fi.get_turbine_power()[1] / fi.get_turbine_power()[0]


    ax_rat.plot(sweep_locations, power_out, color=color,label=label)
    ax_rat.legend()
    ax_rat.grid(True)
    ax_rat.set_title("Normalized Power at %.1f D downstream" % x_loc_1)
    ax_rat.set_ylabel('Normalized Power (-)')
    ax_rat.set_xlabel('Cross-Stream Location (D)')

st.write(fig_cut)
st.write(fig_rat)




########### SINGLE WAKE CASE LONG

st.write("# Single Wake Long wise")

# Determine sweep location
y_loc_1 = 0
sweep_loc = y_loc_1 * D

# Calibrate models to single wake
fig_rat, ax_rat = plt.subplots(1,1)
for i in range(num_models):

    # Grab the model info
    fi = fi_list[i]
    label = label_list[i]
    color = color_list[i]

    # Configure model
    fi.reinitialize_flow_field(
        wind_speed=ws, layout_array=[[0], [0]]
    )
    fi.calculate_wake(yaw_angles=[yaw_1])

    # Perform the sweep (cross wind)
    sweep_locations = np.arange(2, 40, 0.5)
    power_out = np.zeros_like(sweep_locations)

    for x_idx, x_loc in enumerate(sweep_locations):

        fi.reinitialize_flow_field(
            layout_array=[[0,x_loc * D],
                [0, 0]]
        )

        fi.calculate_wake(yaw_angles=[0,0])

        power_out[x_idx] = fi.get_turbine_power()[1] / fi.get_turbine_power()[0]


    ax_rat.plot(sweep_locations, power_out, color=color,label=label)
    ax_rat.legend()
    ax_rat.grid(True)
    ax_rat.set_title("Normalized Power at %.1f D downstream" % x_loc_1)
    ax_rat.set_ylabel('Normalized Power (-)')
    ax_rat.set_xlabel('Cross-Stream Location (D)')


st.write(fig_rat)


########### Double WAKE CASE

st.write("# Double Wake")


# Determine locations
t1_x_loc =  spacing_2 * D 
t1_y_loc =  offset_2 * D

sweep_loc = t1_x_loc + x_loc_2 * D

# Calibrate models to single wake
fig_cut, axarr_cut = plt.subplots(1,3)
fig_rat, ax_rat = plt.subplots(1,1)
for i in range(num_models):

    # Grab the model info
    fi = fi_list[i]
    label = label_list[i]
    color = color_list[i]

    # Configure model
    fi.reinitialize_flow_field(
        wind_speed=ws, layout_array=[[0, t1_x_loc], [0, t1_y_loc]]
    )
    fi.calculate_wake(yaw_angles=[0,0])

    # Show the hor plane
    ax = axarr_cut[i]
    hor_plane = fi.get_hor_plane()
    wfct.visualization.visualize_cut_plane(
        hor_plane, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
    )
    ax.set_title(label)
    ax.axvline(sweep_loc,color='w',ls='--')
    ax.axhline(0,color='w',ls='--')

    # Perform the sweep

    sweep_locations = np.arange(-3, 3.25, 0.25)
    power_out = np.zeros_like(sweep_locations)

    for y_idx, y_loc in enumerate(sweep_locations):

        fi.reinitialize_flow_field(
            layout_array=(
                [
                    0,
                    t1_x_loc,
                    sweep_loc
                ],
                [0, t1_y_loc, y_loc * D],
            )
        )
        fi.calculate_wake(yaw_angles=[0,0,0])
        power_out[y_idx] = fi.get_turbine_power()[2] / fi.get_turbine_power()[0]


    ax_rat.plot(sweep_locations, power_out, color=color,label=label)
    ax_rat.legend()
    ax_rat.grid(True)
    ax_rat.set_title("Normalized Power at %.1f D downstream" % x_loc_1)
    ax_rat.set_ylabel('Normalized Power (-)')
    ax_rat.set_xlabel('Cross-Stream Location (D)')

st.write(fig_cut)
st.write(fig_rat)

# # Options for deep array case
# x_loc_3 = st.sidebar.slider("Deep: D downstream", 2., 50., 7.0, step=1.0)
# n_row = st.sidebar.slider("Deep: Num Rows", 2., 12., 7.0, step=1.0)
# spacing_3 = st.sidebar.slider("Deep: D Spacing", 2., 12., 7.0, step=1.0)
# offset_3 = st.sidebar.slider("Deep: D Offset", -2., 2., 0.0, step=0.25)


########### Double WAKE CASE

st.write("# Deep Array Wake")

# Calculate turbines
deep_x = []
deep_y = []
for i in range(n_row):
    for j in range(n_row):
        deep_x.append(i*D*spacing_3)
        deep_y.append(j*D*spacing_3 + i*D*offset_3)

deep_x = np.array(deep_x)
deep_y = np.array(deep_y)

# Determine locations
sweep_loc = np.max(deep_x) + x_loc_3 * D
sweep_center = np.mean(np.array(deep_y))

# Calibrate models to single wake
fig_cut, axarr_cut = plt.subplots(1,3)
fig_rat, ax_rat = plt.subplots(1,1)
for i in range(num_models):

    # Grab the model info
    fi = fi_list[i]
    label = label_list[i]
    color = color_list[i]

    # Configure model
    fi.reinitialize_flow_field(
        wind_speed=ws, layout_array=[deep_x, deep_y]
    )
    fi.calculate_wake(yaw_angles=np.zeros_like(deep_x))

    # Show the hor plane
    ax = axarr_cut[i]
    hor_plane = fi.get_hor_plane()
    wfct.visualization.visualize_cut_plane(
        hor_plane, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
    )
    ax.set_title(label)
    ax.axvline(sweep_loc,color='w',ls='--')
    ax.axhline(sweep_center,color='w',ls='--')

    # Perform the sweep

    # sweep_locations = np.arange(np.min(deep_y) - 2, np.max(deep_y) + 2.5, 0.25)
    sweep_locations = np.linspace(np.min(deep_y)/D - 2, np.max(deep_y)/D + 2.5, 20)
    power_out = np.zeros_like(sweep_locations)

    x_locs = [xx for xx in deep_x] + [sweep_loc]
    y_locs = [xx for xx in deep_y] + [0]

    for y_idx, y_loc in enumerate(sweep_locations):

        print(y_idx, y_loc)
        y_locs[-1] = y_loc * D
        print(y_locs)
        fi.reinitialize_flow_field(
            layout_array=(
                x_locs,
                y_locs
            )
        )
        fi.calculate_wake(yaw_angles=np.zeros(len(deep_x)+1))
        power_out[y_idx] = fi.get_turbine_power()[-1] / fi.get_turbine_power()[0]


    ax_rat.plot(sweep_locations, power_out, color=color,label=label)
    ax_rat.legend()
    ax_rat.grid(True)
    ax_rat.set_title("Normalized Power at %.1f D downstream" % x_loc_1)
    ax_rat.set_ylabel('Normalized Power (-)')
    ax_rat.set_xlabel('Cross-Stream Location (D)')

st.write(fig_cut)
st.write(fig_rat)
