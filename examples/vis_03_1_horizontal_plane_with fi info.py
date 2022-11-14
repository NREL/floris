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

fi = FlorisInterface("floris/examples/inputs/gch.yaml")

# # Define 4 turbines
layout_x = np.array([3000.0, 0.0, 1500.0, 3000.0])
layout_y = np.array([800.0, 800.0, 800.0, 0.0])
if 0 : turbine_type = ['nrel_5MW', 'nrel_5MW', 'nrel_5MW', 'nrel_5MW'] # same WTGs
if 1 : turbine_type = ['nrel_5MW', 'nrel_5MW', 'iea_10MW', 'iea_15MW'] # mix WTGs
fi.reinitialize(layout_x=layout_x, layout_y=layout_y, turbine_type=turbine_type)

# plot with yawangle, fi info
if 1 : # dh. 화면에 yaw, fi 정보도 출력
    # wind directions and speeds for plot
    wd = [[i] for i in np.arange(45,360,90)];  
    ws = [8.0]; 
    if 0: ws = [[i] for i in np.arange(3,25,1.0)]; 
    
    # yaw angles: Change the yaw angles and configure the plot differently
    n_wd=fi.floris.flow_field.n_wind_directions; n_ws=fi.floris.flow_field.n_wind_speeds
    n_wtg=fi.floris.farm.n_turbines
    yaw_angles = np.zeros((1, 1, n_wtg));
    yaw_angles[:,:,:] = (0, 0, 15, -15)

    # ready for plot 
    n_col=2
    fig, ax_list = plt.subplots( round(len(wd)/n_col+0.5), n_col, figsize=(16, 8))
    ax_list = ax_list.flatten()
    res=200;
    for i in range(len(wd)):
        # fi update for text on WTGs.
        # at the end of fi.calculate_horizontal_plane, restoring fi to previous,
        # we can't use those results for texting
        fi.reinitialize( wind_speeds=ws, wind_directions=wd[i] ) # class fi  
        fi.floris.farm.yaw_angles=yaw_angles; # yaw angles
        
        # getting df (x,y,z, u,v,w) for planar flow field
        horizontal_plane=fi.calculate_horizontal_plane(wd=wd[i], ws=ws, height=90.0, yaw_angles=yaw_angles, x_resolution=res, y_resolution=res)

        # plot
        ax=ax_list[i];
        visualize_cut_plane(horizontal_plane, ax=ax, title="Wind direction "+str(wd[i])+"deg", color_bar=True);
        plot_turbines_with_fi(ax=ax_list[i],fi=fi)  # , wd=wd[i] 
        FarmP = fi.get_farm_power()/1000 # 이거??
        plt.xlabel(f'{FarmP[0,0]:.3f}'+' KW')
        
        #dh. text on WTGs
        # fi update with reinitialize
        turbine_yaw = fi.floris.farm.yaw_angles
        turbine_type= fi.floris.farm.turbine_type
        turbine_avg_vel=fi.get_turbine_average_velocities()
        turbine_powers = fi.get_turbine_powers()/1000.
        turbine_ais =fi.get_turbine_ais()
        turbine_Cts = fi.get_turbine_Cts()

        mytext = [f"yaw: {i:.1f}" for i in turbine_yaw[0,0]] 
        if 1: mytext = [f"T{i}: {turbine_type[i]} \n {mytext[i]}" for i in range(n_wtg)] 
        if 1: mytext = [f"{mytext[i]} \n Vel: {turbine_avg_vel[0,0,i]:.1f}" for i in range(n_wtg)] 
        if 1: mytext = [f"{mytext[i]} \n Pow: {turbine_powers[0,0,i]:.1f}" for i in range(n_wtg)] 
        if 0: mytext = [f"{mytext[i]} \n ai: {turbine_ais[0,0,i]:.1f}" for i in range(n_wtg)] 
        if 1: mytext = [f"{mytext[i]} \n ct: {turbine_Cts[0,0,i]:.1f}" for i in range(n_wtg)] 
        for j in range(fi.floris.farm.n_turbines):
            ax.text(fi.layout_x[j], fi.layout_y[j], mytext[j], color='springgreen')

        #dh. text on Farm
        ax.text(min(horizontal_plane.df.x1), min(horizontal_plane.df.x2), f' FarmPower: {FarmP[0,0]:.2f}'+f' KW \n WD: {wd[i]}, WS: {ws[0]} \n',color='white')
        
    plt.tight_layout(); plt.savefig("abc.png"); plt.show(); 