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

from floris.tools import FlorisInterface
from floris.tools.visualization import visualize_cut_plane
from floris.tools.visualization import plot_rotor_values

#dh
import numpy as np
from floris.tools.visualization import plot_turbines_with_fi
from floris.utilities import rotate_coordinates_rel_west
import math #dh. for drawing swept area

fi = FlorisInterface("floris/examples/inputs/gch.yaml")
flow_field=fi.floris.flow_field
farm=fi.floris.farm
grid=fi.floris.grid
n_WTG=farm.n_turbines; 
wd = flow_field.wind_directions
ws = flow_field.wind_speeds

# check : Farm layout and flowfield info
if 1 : 
    fig, ax = plt.subplots(1, 1)    
    plot_turbines_with_fi(ax=ax,fi=fi)

    # text on WTGs
    turbine_type= farm.turbine_type_names_sorted    
    mytext = [f"yaw: {i:.1f}" for i in farm.yaw_angles[0,0]] # after FI, yaw_angles are set 0
    if 1: mytext = [f"T{i}: {turbine_type[i]} \n {mytext[i]}" for i in range(n_WTG)] 
    for j in range(n_WTG):
        ax.text(fi.layout_x[j], fi.layout_y[j], mytext[j], color='b')

    # text on Farm
    mytext = [f" WD: {wd}, WS: {ws}"] # height
    if 1: mytext = [f"{mytext} \n TI: {flow_field.turbulence_intensity}, shear: {flow_field.wind_shear}, veer: {flow_field.wind_veer}"] 
    ax.text(min(fi.layout_x), min(fi.layout_x), mytext, color='black')
    
    #plt.tight_layout(); plt.show()

# check : Turbine Grid for 1 wind direction rel to west
if 1 : 
    fig, ax = plt.subplots(1, 1)       
    ax = fig.add_subplot(projection='3d')
    
    xWTG, y_WTG, z_WTG=rotate_coordinates_rel_west(np.array([wd[0]]), (grid.x_sorted[0,0], grid.y_sorted[0,0], grid.z_sorted[0,0]), inv_rot=True)
    ax.scatter(xWTG,y_WTG,z_WTG) # 작동
    
    # text on WTGs
    turbine_type= farm.turbine_type_names_sorted
    hh=farm.hub_heights; RD=farm.rotor_diameters
    mytext = [f"yaw: {i:.1f}" for i in farm.yaw_angles[0,0]] # after FI, yaw_angles are set 0
    if 1: mytext = [f"T{i}: {turbine_type[i]} \n {mytext[i]}" for i in range(n_WTG)] 
    if 1: mytext = [f"{mytext[i]} \n HH: {hh[i]}" for i in range(n_WTG)] 
    if 1: mytext = [f"{mytext[i]} \n RD: {RD[i]}" for i in range(n_WTG)] 
    for j in range(n_WTG):
        ax.text(fi.layout_x[j], fi.layout_y[j], farm.hub_heights[j], mytext[j], color='b')


if 1: # change some information
    # Define 4 turbines
    layout_x = np.array([3000.0, 0.0, 1500.0, 3000.0])
    layout_y = np.array([800.0, 800.0, 800.0, 0.0])
    if 0 : turbine_type = ['nrel_5MW', 'nrel_5MW', 'nrel_5MW', 'nrel_5MW'] # same WTGs
    if 1 : turbine_type = ['nrel_5MW', 'nrel_5MW', 'iea_10MW', 'iea_15MW'] # mix WTGs
    fi.reinitialize(layout_x=layout_x, layout_y=layout_y, turbine_type=turbine_type)
    
    flow_field=fi.floris.flow_field; farm=fi.floris.farm; grid=fi.floris.grid
    n_WTG=farm.n_turbines
    wd = flow_field.wind_directions
    ws = flow_field.wind_speeds

    # set yaw angles
    n_WTG = 4
    yaw_angles = np.zeros((1, 1, n_WTG))
    yaw_angles[:,:,:] = (0, 0, 15, -15)
    fi.floris.farm.yaw_angles=yaw_angles; # update yaw in fi class
    
# check : Farm layout and flowfield info
if 1 : 
    fig, ax = plt.subplots(1, 1)    
    plot_turbines_with_fi(ax=ax,fi=fi)

    # text on WTGs
    turbine_type= farm.turbine_type_names_sorted    
    mytext = [f"yaw: {i:.1f}" for i in farm.yaw_angles[0,0]] # after FI, yaw_angles are set 0
    if 1: mytext = [f"T{i}: {turbine_type[i]} \n {mytext[i]}" for i in range(n_WTG)] 
    for j in range(n_WTG):
        ax.text(fi.layout_x[j], fi.layout_y[j], mytext[j], color='b')

    # text on Farm
    mytext = [f" WD: {wd}, WS: {ws}"] # height
    if 1: mytext = [f"{mytext} \n TI: {flow_field.turbulence_intensity}, shear: {flow_field.wind_shear}, veer: {flow_field.wind_veer}"] 
    ax.text(min(fi.layout_x), min(fi.layout_x), mytext, color='black')
    
    #plt.tight_layout(); plt.show()

# check : Turbine Grid for 1 wind direction rel to west
if 1 : 
    fig, ax = plt.subplots(1, 1)       
    ax = fig.add_subplot(projection='3d')
    
    xWTG, y_WTG, z_WTG=rotate_coordinates_rel_west(np.array([wd[0]]), (grid.x_sorted[0,0], grid.y_sorted[0,0], grid.z_sorted[0,0]), inv_rot=True)
    ax.scatter(xWTG,y_WTG,z_WTG) # 작동
    
    # text on WTGs
    turbine_type= farm.turbine_type_names_sorted
    hh=farm.hub_heights; RD=farm.rotor_diameters
    mytext = [f"yaw: {i:.1f}" for i in farm.yaw_angles[0,0]] # after FI, yaw_angles are set 0
    if 1: mytext = [f"T{i}: {turbine_type[i]} \n {mytext[i]}" for i in range(n_WTG)] 
    if 1: mytext = [f"{mytext[i]} \n HH: {hh[i]}" for i in range(n_WTG)] 
    if 1: mytext = [f"{mytext[i]} \n RD: {RD[i]}" for i in range(n_WTG)] 
    for j in range(n_WTG):
        ax.text(fi.layout_x[j], fi.layout_y[j], farm.hub_heights[j], mytext[j], color='b')
        
        
plt.tight_layout(); plt.show()