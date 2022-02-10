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
import streamlit as st
import numpy as np

from floris.tools import FlorisInterface
from floris.tools.visualization import visualize_cut_plane


# """
# This example demonstrates an interactive visual comparison of FLORIS
# wake models using streamlit

# To run this example:
# (with your FLORIS environment enabled)
# pip install streamlit

# streamlit run 12_streamlit_demo.py
# """

# I think this example shows some interesting things
# 1) Something is odd with the Jensen model
# 2) Doing reinitialize in a loop without redoing the interface gives an error (try it!)
# 3) CC can't be visualized

# Parameters
wind_speed = 8.0
# ti = 0.06

# Set to wide
st.set_page_config(layout="wide")

#Streamlit inputs
wind_direction_user = st.sidebar.slider("Wind Direction", 240., 300., 285., step=1.)
spacing = st.sidebar.slider("Turbine spacing (D)", 3., 10., 6., step=0.5)
N = st.sidebar.slider("Turbines per row", 4, 8, 5, step=1)



# Get number of turbines and make 0 yaw angle matrix
num_turbine = N**3
yaw_angles = np.zeros((1, 1, num_turbine)) # 1 wd/ 1ws/ N*N turbines

# Grab two different wake models
fi_gch = FlorisInterface("inputs/gch.yaml")


# Define layout
X, Y = np.meshgrid(
    spacing * fi_gch.floris.turbine.rotor_diameter * np.arange(0, N, 1),
    spacing * fi_gch.floris.turbine.rotor_diameter * np.arange(0, N, 1),
)
X = X.flatten()
Y = Y.flatten()


# Title
st.title('FLORIS Model Comparison')

# Create the main analysis image
fig, axarr = plt.subplots(2,2)

# Calculate for alligned case and user requested direction
for wd_idx, wind_direction in enumerate([270., wind_direction_user]):

    # Grab two different wake models
    # Not positive why this is necessary 
    # If you move this above the loop so it only happens once, there is an error
    fi_jensen = FlorisInterface("inputs/jensen.yaml")
    fi_gch = FlorisInterface("inputs/gch.yaml")
    fi_cc = FlorisInterface("inputs/cc.yaml")
    
    # Configure model
    fi_jensen.reinitialize( layout=( X, Y ), wind_speeds=[wind_speed], wind_directions=[wind_direction], turbulence_intensity=0.05 )
    fi_gch.reinitialize( layout=( X, Y ), wind_speeds=[wind_speed], wind_directions=[wind_direction], turbulence_intensity=0.06 )
    fi_cc.reinitialize( layout=( X, Y ), wind_speeds=[wind_speed], wind_directions=[wind_direction], turbulence_intensity=0.1)

    # Calculate wake
    fi_jensen.calculate_wake(yaw_angles=yaw_angles)
    fi_gch.calculate_wake(yaw_angles=yaw_angles)
    fi_cc.calculate_wake(yaw_angles=yaw_angles)

    # Get turbine powers
    turbine_powers_jensen = fi_jensen.get_turbine_powers()
    turbine_powers_gch = fi_gch.get_turbine_powers()
    turbine_powers_cc = fi_cc.get_turbine_powers()

    # Put the turbine powers in descending order
    turbine_powers_jensen = np.sort(turbine_powers_jensen.flatten())[::-1]/1000.
    turbine_powers_gch = np.sort(turbine_powers_gch.flatten())[::-1]/1000.
    turbine_powers_cc = np.sort(turbine_powers_cc.flatten())[::-1]/1000.

    # Show the (GCH) Horizontal plane
    horizontal_plane_gch = fi_gch.get_hor_plane(x_resolution=100, y_resolution=100)
    ax = axarr[0,wd_idx]
    visualize_cut_plane(horizontal_plane_gch, ax=ax, title="Wind Direction = %.1f" % wind_direction)

    # Compare the power production
    ax = axarr[1,wd_idx]
    ax.plot(turbine_powers_jensen,color='k',label='Jensen')
    ax.plot(turbine_powers_gch,color='b',label='GCH')
    ax.plot(turbine_powers_cc,ls='--',color='r',label='CC')
    ax.grid(True)
    ax.legend()
    ax.set_ylabel('Power (kW)')
    ax.set_xlabel('Turbine (sorted)')

# Show the figure
st.write(fig)
