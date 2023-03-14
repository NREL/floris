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


import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from floris.tools import FlorisInterface
from floris.tools.visualization import visualize_cut_plane


# import seaborn as sns



# """
# This example demonstrates an interactive visual comparison of FLORIS
# wake models using streamlit

# To run this example:
# (with your FLORIS environment enabled)
# pip install streamlit

# streamlit run 16_streamlit_demo.py
# """


# Parameters
wind_speed = 8.0
# ti = 0.06

# Set to wide
st.set_page_config(layout="wide")

# Parameters
D = 126. # Assume for convenience
floris_model_list = ['jensen','gch','cc','turbopark']
color_dict = {
    'jensen':'k',
    'gch':'b',
    'cc':'r',
    'turbopark':'c'
}

# Streamlit inputs
n_turbine_per_row = st.sidebar.slider("Turbines per row", 1, 8, 2, step=1)
n_row = st.sidebar.slider("Number of rows", 1, 8,1, step=1)
spacing = st.sidebar.slider("Turbine spacing (D)", 3., 10., 6., step=0.5)
wind_direction = st.sidebar.slider("Wind Direction", 240., 300., 270., step=1.)
wind_speed = st.sidebar.slider("Wind Speed", 4., 15., 8., step=0.25)
turbulence_intensity = st.sidebar.slider("Turbulence Intensity", 0.01, 0.25, 0.06, step=0.01)
floris_models = st.sidebar.multiselect("FLORIS Models", floris_model_list, floris_model_list)
# floris_models_viz = st.sidebar.multiselect(
#     "FLORIS Models for Visualization",
#     floris_model_list,
#     floris_model_list
# )
desc_yaw = st.sidebar.checkbox("Descending yaw pattern?",value=False)
front_turbine_yaw = st.sidebar.slider("Upstream yaw angle", -30., 30., 0., step=0.5)

# Define the layout
X = []
Y = []

for x_idx in range(n_turbine_per_row):
    for y_idx in range(n_row):
        X.append(D * spacing * x_idx)
        Y.append(D * spacing * y_idx)

turbine_labels = ['T%02d' % i for i in range(len(X))]

# Set up the yaw angle values
yaw_angles_base = np.zeros([1,1,len(X)])

yaw_angles_yaw = np.zeros([1,1,len(X)])
if not desc_yaw:
    yaw_angles_yaw[:,:,:n_row] = front_turbine_yaw
else:
    decreasing_pattern = np.linspace(front_turbine_yaw,0,n_turbine_per_row)
    for i in range(n_turbine_per_row):
        yaw_angles_yaw[:,:,i*n_row:(i+1)*n_row] = decreasing_pattern[i]



# Get a few quanitities
num_models = len(floris_models)

# Determine which models to plot given cant plot cc right now
floris_models_viz = [m for m in floris_models if "cc" not in m]
floris_models_viz = [m for m in floris_models_viz if "turbo" not in m]
num_models_to_viz = len(floris_models_viz)

# Set up the visualization plot
fig_viz, axarr_viz = plt.subplots(num_models_to_viz,2)

# Set up the turbine power plot
fig_turb_pow, ax_turb_pow = plt.subplots()

# Set up a list to save the farm power results
farm_power_results = []

# Now complete all these plots in a loop
for fm in floris_models:

    # Analyze the base case==================================================
    print('Loading: ',fm)
    fi = FlorisInterface("inputs/%s.yaml" % fm)

    # Set the layout, wind direction and wind speed
    fi.reinitialize(
        layout_x=X,
        layout_y=Y,
        wind_speeds=[wind_speed],
        wind_directions=[wind_direction],
        turbulence_intensity=turbulence_intensity
    )

    fi.calculate_wake(yaw_angles=yaw_angles_base)
    turbine_powers = fi.get_turbine_powers() / 1000.
    ax_turb_pow.plot(
        turbine_labels,
        turbine_powers.flatten(),
        color=color_dict[fm],
        ls='-',
        marker='s',
        label='%s - baseline' % fm
    )
    ax_turb_pow.grid(True)
    ax_turb_pow.legend()
    ax_turb_pow.set_xlabel('Turbine')
    ax_turb_pow.set_ylabel('Power (kW)')

    # Save the farm power
    farm_power_results.append((fm,'base',np.sum(turbine_powers)))

    # If in viz list also visualize
    if fm in floris_models_viz:
        ax_idx = floris_models_viz.index(fm)
        ax = axarr_viz[ax_idx, 0]

        horizontal_plane_gch = fi.calculate_horizontal_plane(
            x_resolution=100,
            y_resolution=100,
            yaw_angles=yaw_angles_base,
            height=90.0
        )
        visualize_cut_plane(horizontal_plane_gch, ax=ax, title='%s - baseline' % fm)

    # Analyze the yawed case==================================================
    print('Loading: ',fm)
    fi = FlorisInterface("inputs/%s.yaml" % fm)

    # Set the layout, wind direction and wind speed
    fi.reinitialize(
        layout_x=X,
        layout_y=Y,
        wind_speeds=[wind_speed],
        wind_directions=[wind_direction],
        turbulence_intensity=turbulence_intensity
    )

    fi.calculate_wake(yaw_angles=yaw_angles_yaw)
    turbine_powers = fi.get_turbine_powers() / 1000.
    ax_turb_pow.plot(
        turbine_labels,
        turbine_powers.flatten(),
        color=color_dict[fm],
        ls='--',
        marker='o',
        label='%s - yawed' % fm
    )
    ax_turb_pow.grid(True)
    ax_turb_pow.legend()
    ax_turb_pow.set_xlabel('Turbine')
    ax_turb_pow.set_ylabel('Power (kW)')

    # Save the farm power
    farm_power_results.append((fm,'yawed',np.sum(turbine_powers)))

    # If in viz list also visualize
    if fm in floris_models_viz:
        ax_idx = floris_models_viz.index(fm)
        ax = axarr_viz[ax_idx, 1]

        horizontal_plane_gch = fi.calculate_horizontal_plane(
            x_resolution=100,
            y_resolution=100,
            yaw_angles=yaw_angles_yaw,
            height=90.0
        )
        visualize_cut_plane(horizontal_plane_gch, ax=ax, title='%s - yawed' % fm)

st.header("Visualizations")
st.write(fig_viz)
st.header("Power Comparison")
st.write(fig_turb_pow)
