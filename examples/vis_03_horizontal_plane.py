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

fi = FlorisInterface("floris/examples/inputs/gch.yaml") # 3.2.1.2.1.1

# # Define 4 turbines
layout_x = np.array([3000.0, 0.0, 1500.0, 3000.0])
layout_y = np.array([800.0, 800.0, 800.0, 0.0])
if 1 : turbine_type = ['nrel_5MW', 'nrel_5MW', 'nrel_5MW', 'nrel_5MW'] # same WTGs
if 0 : turbine_type = ['nrel_5MW', 'nrel_5MW', 'iea_10MW', 'iea_15MW'] # mix WTGs
fi.reinitialize(layout_x=layout_x, layout_y=layout_y, turbine_type=turbine_type)

# sweep_wind_directions
# just with yawangle
if 1 : # just with yaw angle text
    # select wind directions and wind speed for horizontal plot
    wd = [[i] for i in np.arange(0,360,90)];  #change : wind directions
    ws = [8.0]
    # yaw angles: Change the yaw angles and configure the plot differently
    n_wd=fi.floris.flow_field.n_wind_directions; n_ws=fi.floris.flow_field.n_wind_speeds
    n_wtg=fi.floris.farm.n_turbines
    yaw_angles = np.zeros((1, 1, n_wtg));
    yaw_angles[:,:,:] = (0, 0, 15, -15)
    
    # ready for plot
    n_col=2 #change : graph's column
    fig, ax_list = plt.subplots( round(len(wd)/n_col+0.5), n_col, figsize=(16, 8))
    ax_list = ax_list.flatten()
    
    horizontal_plane =[]; res=200;
    # get DFs (x,y,z, u,v,w) for horizontal plane
    for i in range(len(wd)):
        horizontal_plane.append(fi.calculate_horizontal_plane(wd=wd[i], ws=ws, height=90.0, yaw_angles=yaw_angles, x_resolution=res, y_resolution=res), )

    # plot DFs
    for i in range(len(wd)):
        ax=ax_list[i];
        visualize_cut_plane(horizontal_plane[i], ax=ax, title="Wind direction "+str(wd[i])+"deg", color_bar=True);
        
        # text on WTGs
        turbine_yaw = yaw_angles.flatten()
        turbine_type= fi.floris.farm.turbine_type
        
        mytext = [f"yaw: {i:.1f}" for i in turbine_yaw] 
        if 1: mytext = [f"T{i:0d}: {turbine_type[i]} \n {mytext[i]}" for i in range(n_wtg)] 
        for j in range(fi.floris.farm.n_turbines):
            ax.text(fi.layout_x[j], fi.layout_y[j], mytext[j], color='springgreen')
        
        # text on Farm
        ax.text(min(horizontal_plane[i].df.x1), min(horizontal_plane[i].df.x2), f' WD: {wd[i]}, WS: {ws[0]} \n',color='white')
        
    plt.tight_layout(); plt.savefig("fix_orient.png"); plt.show(); 