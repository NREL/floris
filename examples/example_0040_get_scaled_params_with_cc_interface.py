#
# Copyright 2019 NREL
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#

# Try out variation of D and rating

import floris.tools as wfct
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pickle
import floris.tools.cc_blade_utilities as ccb
from ccblade import CCAirfoil, CCBlade

# Some useful constants
degRad = np.pi/180.
rpmRadSec = 2.0*(np.pi)/60.0

# Load the sowfa case for an example turbine input file for the NREL 5MW
sowfa_case = wfct.sowfa_utilities.SowfaInterface('sowfa_example')

# Grab the turbine dict in order to have controller values
turbine_dict = wfct.sowfa_utilities.read_foam_file(os.path.join(sowfa_case.case_folder, sowfa_case.turbine_sub_path, sowfa_case.turbine_name))

# Select an R and rating to test
# Get a scaled version to 40m R, 2MW
turbine_dict_base, rotor_base = ccb.scale_controller_and_rotor(turbine_dict)
turbine_dict_scaled_2mw, rotor_scaled_2mw = ccb.scale_controller_and_rotor(turbine_dict,40,2)

# Compare the torque curves
fig, ax = plt.subplots()
ccb.show_torque_curve(turbine_dict_base,ax,label='Baseline')
ccb.show_torque_curve(turbine_dict_scaled_2mw,ax,label='Scaed')

# Generate the baseline lut
# if it doesn't yet exist
if not os.path.exists('cp_ct_cq_lut.p'):
    ccb.generate_base_lut(rotor_base, turbine_dict_base)

# Check the steady solutions at 9 m/s
ws = 9.
ccb.get_steady_state(turbine_dict_base,rotor_base,ws,title='Baseline',show_plot=True)
ccb.get_steady_state(turbine_dict_scaled_2mw,rotor_scaled_2mw,ws,title='Scaled',show_plot=True)


# Now get the full curves
ws_array = np.arange(5,25,2.)
ws_array, pow_array_base, cp_array_base,ct_array_base = ccb.get_wind_sweep_steady_values(turbine_dict_base,rotor_base,ws_array=ws_array)
ws_array, pow_array_scaled, cp_array_scaled,ct_array_scaled = ccb.get_wind_sweep_steady_values(turbine_dict_scaled_2mw,rotor_scaled_2mw,ws_array=ws_array)

# Show these curvess
fig, axarr = plt.subplots(3,1,sharex=True)

ax = axarr[0]
ax.plot(ws_array, pow_array_base/1E6, label='Baseline')
ax.plot(ws_array, pow_array_scaled/1E6, label='Scaled')
ax.set_ylabel('Power')
ax.grid(True)
ax.legend()

ax = axarr[1]
ax.plot(ws_array, cp_array_base, label='Baseline')
ax.plot(ws_array, cp_array_scaled, label='Scaled')
ax.set_ylabel('Cp')
ax.grid(True)
ax.legend()

ax = axarr[2]
ax.plot(ws_array, ct_array_base, label='Baseline')
ax.plot(ws_array, ct_array_scaled, label='Scaled')
ax.set_ylabel('Ct')
ax.grid(True)
ax.legend()

# Finally output both in a format good for setting FLORIS
print("baseline")
print(repr(ws_array))
print(repr(cp_array_base * turbine_dict_base['GenEfficiency']))
print(repr(ct_array_base))

print("scaled")
print(repr(ws_array))
print(repr(cp_array_scaled * turbine_dict_scaled_2mw['GenEfficiency']))
print(repr(ct_array_scaled))

plt.show()
