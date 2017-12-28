"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import sys
sys.path.append('../floris')
from floris import floris
import OptModules
import numpy as np
from scipy.optimize import minimize

floris = floris()
floris.process_input("floris.json")

floris.farm.flow_field.plot_flow_field_planes()

# %%  
minimum_yaw_angle = 0.0
maxiumum_yaw_angle = 20.0

# list comprehension (turbine_map())
# var = [i for i in [1,2,3]]

# var = []
# for i in [1,2,3]:
# 	var.append(i)

x0 = []
bnds = []
power0 = 0
for i,coord in enumerate(floris.farm.get_turbine_coords()):
    turbine = floris.farm.get_turbine_at_coord(coord)
    x0.append(turbine.yaw_angle)
    bnds.append((np.radians(minimum_yaw_angle),np.radians(maxiumum_yaw_angle)))
    power0 = power0 + turbine.power

print('=====================================================================')
print('Optimizing wake redirection control...')
print('Number of parameters to optimize = ', len(x0))
print('=====================================================================')

resPlant = minimize(OptModules.optPlant,x0,args=(floris),method='SLSQP',bounds=bnds,options={'ftol':0.001,'eps':0.05})
print(resPlant)

# %%
yawOpt = resPlant.x

print('Optimal yaw angles for:')
for i in range(len(yawOpt)):
	print('Turbine ', i, ' yaw angle = ', np.degrees(resPlant.x[i]))
    
powerOpt = 0
for i,coord in enumerate(floris.farm.get_turbine_coords()):
    turbine = floris.farm.get_turbine_at_coord(coord)
    turbine.yaw_angle = yawOpt[i]
    powerOpt = powerOpt + turbine.power
floris.farm.flow_field.plot_flow_field_planes()

print('Power increased by ', 100*(powerOpt-power0)/power0)



