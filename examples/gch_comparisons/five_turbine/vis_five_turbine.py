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
 

"""
Compare to the visuzlizations of a 5 turbine case with and without GCH
"""

import matplotlib.pyplot as plt
import floris.tools as wfct
import numpy as np
import pandas as pd
import copy

# Parameter
use_nominal_values = True

# Load the paper results
df_results = pd.read_csv('paper_results_5_turbine.csv')

# # Initialize the FLORIS interface fi
fi_gl = wfct.floris_interface.FlorisInterface("../../example_input.json")


# Select gauss legacy with GCH off
# fi_gl.floris.farm.set_wake_model('gauss_legacy')
fi_gl.set_gch(False) # Disable GCH
# fi_gl.floris.farm.set_wake_model('gauss_merge')

# # Match the layout
x_layout = tuple(map(
    float,
    df_results.layout_x.values[0].replace('(','').replace(')','').split(',')
))
y_layout = tuple(map(
    float,
    df_results.layout_y.values[0].replace('(','').replace(')','').split(',')
))
fi_gl.reinitialize_flow_field(layout_array=[x_layout,y_layout])

# Match the inflow
U0 = df_results.floris_U0.values[0]
TI = df_results.floris_TI.values[0]
fi_gl.reinitialize_flow_field(wind_speed=U0,turbulence_intensity=TI)

# Set up the gch model
fi_gch = copy.deepcopy(fi_gl)
fi_gch.set_gch(True)

fig, axarr = plt.subplots(2,2,sharex=True, sharey=True,figsize=(14,3))

# Legacy baseline
ax = axarr[0,0]
fi = fi_gl
fi.calculate_wake([0,0,0,0,0])
hor_plane = fi.get_hor_plane()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
wfct.visualization.plot_turbines_with_fi(ax,fi)
ax.set_title("Legacy Aligned")
print(fi.get_farm_power())


# Legacy yawed
ax = axarr[0,1]
fi = fi_gl
fi.calculate_wake([25,15,0,0,0])
hor_plane = fi.get_hor_plane()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
ax.set_title("Legacy Yawed")
wfct.visualization.plot_turbines_with_fi(ax,fi)
print(fi.get_farm_power())

# GCH baseline
ax = axarr[1,0]
fi = fi_gch
fi.calculate_wake([0,0,0,0,0])
hor_plane = fi.get_hor_plane()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
wfct.visualization.plot_turbines_with_fi(ax,fi)
ax.set_title("GCH Aligend")
print(fi.get_farm_power())


# GCH yawed
ax = axarr[1,1]
fi = fi_gch
fi.calculate_wake([25,15,0,0,0])
hor_plane = fi.get_hor_plane()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
ax.set_title("GCH Yawed")
wfct.visualization.plot_turbines_with_fi(ax,fi)
print(fi.get_farm_power())


plt.show()
