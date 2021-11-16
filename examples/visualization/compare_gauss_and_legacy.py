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
#    streamlit run interactive_visual.py

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import floris.tools as wfct


# Fixed parameters
minSpeed = 6.0
maxSpeed = 8.0
D = 126.0
HH = 90.0

# Options
ws = st.sidebar.slider("Wind Speed", 5.0, 10.0, 8.0, step=0.1)
wd = st.sidebar.slider("Wind Direction", 250, 290, 270, step=2)
yaw_1 = st.sidebar.slider("Yaw angle T1", -30, 30, 0, step=1)
x_loc = st.sidebar.slider("x normal plane intercept", 0.0, 35.0, 7.0, step=0.25)
y_loc = st.sidebar.slider("y normal plane intercept", -2.0, 2.0, 0.0, step=0.1)
second_turbine = st.sidebar.checkbox("Second Turbine?")
GCH = st.sidebar.checkbox("GCH")


# Initialize the FLORIS interface fi (legacy)
fi = wfct.floris_interface.FlorisInterface("../example_input.json")
fi.set_gch(GCH)
if second_turbine:
    fi.reinitialize_flow_field(
        wind_speed=ws,
        wind_direction=wd,
        layout_array=((0, 126 * 7), (0, 0))
        # wind_speed = ws, wind_direction=wd, layout_array=((0, 126 * 7, 126 * 14, 126 * 21, 126 * 28), (0, 0, 0, 0, 0))
    )
    fi.calculate_wake(yaw_angles=[yaw_1, 0])
    # fi.calculate_wake(yaw_angles=[yaw_1, 0, 0, 0, 0])
else:
    fi.reinitialize_flow_field(
        wind_speed=ws, wind_direction=wd, layout_array=([0], [0])
    )
    fi.calculate_wake(yaw_angles=[yaw_1])


# Initialize the FLORIS interface fi (legacy)
fi_gauss = wfct.floris_interface.FlorisInterface("gauss_version.json")
fi_gauss.set_gch(GCH)
if second_turbine:
    fi_gauss.reinitialize_flow_field(
        wind_speed=ws,
        wind_direction=wd,
        layout_array=((0, 126 * 7), (0, 0))
        # wind_speed = ws, wind_direction=wd, layout_array=((0, 126 * 7, 126 * 14, 126 * 21, 126 * 28), (0, 0, 0, 0, 0))
    )
    fi_gauss.calculate_wake(yaw_angles=[yaw_1, 0])
    # fi_gauss.calculate_wake(yaw_angles=[yaw_1, 0, 0, 0, 0])
else:
    fi_gauss.reinitialize_flow_field(
        wind_speed=ws, wind_direction=wd, layout_array=([0], [0])
    )
    fi_gauss.calculate_wake(yaw_angles=[yaw_1, 0])


st.write("# Horizontal planes")

# Horizontal plane

fig, axarr = plt.subplots(2, 1)

hor_plane = fi.get_hor_plane()
ax = axarr[0]
wfct.visualization.visualize_cut_plane(
    hor_plane, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
)
ax.axhline(y_loc * D, color="w", ls="--", lw=1)
ax.axvline(x_loc * D, color="w", ls="--", lw=1)
ax.set_title("Legacy")

hor_plane_gauss = fi_gauss.get_hor_plane()
ax = axarr[1]
wfct.visualization.visualize_cut_plane(
    hor_plane_gauss, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
)
ax.axhline(y_loc * D, color="w", ls="--", lw=1)
ax.axvline(x_loc * D, color="w", ls="--", lw=1)
ax.set_title("Gauss")

st.write(fig)

# print(hor_plane_gauss.df.x2.unique())

# Look at flow center line
x_points = np.arange(-100, D * 16, 1)
y_points = np.ones_like(x_points) * y_loc * D
z_points = np.ones_like(x_points) * HH

# Get the values
flow_points = fi.get_set_of_points(x_points, y_points, z_points).sort_values("x")
flow_points_gauss = fi_gauss.get_set_of_points(
    x_points, y_points, z_points
).sort_values("x")

# Compare wind speeds at centerline
fig, ax = plt.subplots()
ax.plot(flow_points.x / D, flow_points.u, color="b", label="Legacy")
ax.plot(flow_points_gauss.x / D, flow_points_gauss.u, color="g", label="Gauss")
ax.grid(True)
ax.legend()
ax.set_title("Center line wind speed")
st.write(fig)


# Compare the wake profile at x
# Show the velocity profile using a rotor sweep
cross_plane = fi.get_cross_plane(x_loc * D)
x1_locs, v_array = wfct.cut_plane.wind_speed_profile(
    cross_plane, D / 2, HH, resolution=100, x1_locs=None
)
cross_plane_gauss = fi_gauss.get_cross_plane(x_loc * D)
x1_locs, v_array_gauss = wfct.cut_plane.wind_speed_profile(
    cross_plane_gauss, D / 2, HH, resolution=100, x1_locs=None
)

fig, ax = plt.subplots()
ax.plot(x1_locs, v_array, color="b", label="Legacy")
ax.plot(x1_locs, v_array_gauss, color="r", label="Gauss")
ax.grid(True)
ax.set_title("Rotor-averaged velocity profile")
ax.set_ylabel("Rotor-averaged wind speed (m/s)")
ax.set_xlabel("Center of rotor disk (m)")
ax.legend()
st.write(fig)

# # Legacy

# Powers downstream
x_array = np.arange(D, D * 12, D / 2)
leg_res = []
gauss_res = []

for x in x_array:

    fi.reinitialize_flow_field(
        wind_speed=ws, wind_direction=wd, layout_array=([0, x], [0, y_loc * D])
    )
    fi.calculate_wake(yaw_angles=[yaw_1, 0])
    leg_res.append(fi.get_turbine_power()[1])

    fi_gauss.reinitialize_flow_field(
        wind_speed=ws, wind_direction=wd, layout_array=([0, x], [0, y_loc * D])
    )
    fi_gauss.calculate_wake(yaw_angles=[yaw_1, 0])
    gauss_res.append(fi_gauss.get_turbine_power()[1])


fig, ax = plt.subplots()
ax.plot(x_array / D, leg_res, color="b", label="Legacy")
ax.plot(x_array / D, gauss_res, color="r", label="Gauss")
ax.grid(True)
ax.set_title("Power behind turbine")
ax.set_ylabel("Power")
ax.set_xlabel("Distance downstream")
ax.legend()
st.write(fig)
