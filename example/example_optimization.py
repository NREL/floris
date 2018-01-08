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
import warnings

warnings.simplefilter('ignore', RuntimeWarning)

floris = floris()
floris.process_input("floris.json")

floris.farm.flow_field.plot_flow_field_Zplane()

# %%  
minimum_yaw_angle = 0.0
maximum_yaw_angle = 20.0

# set initial conditions
x0 = []
bnds = []

turbines    = [turbine for _, turbine in floris.farm.flow_field.turbine_map.items()]
x0          = [turbine.yaw_angle for turbine in turbines]
bnds        = [(minimum_yaw_angle, maximum_yaw_angle) for turbine in turbines]
power0      = np.sum([turbine.power for turbine in turbines]) 

print('=====================================================================')
print('Optimizing wake redirection control...')
print('Number of parameters to optimize = ', len(x0))
print('=====================================================================')

resPlant = minimize(OptModules.optPlant,x0,args=(floris),method='SLSQP',bounds=bnds,options={'ftol':0.001,'eps':0.05})

# %%
yawOpt = resPlant.x

print('Optimal yaw angles for:')
for i in range(len(yawOpt)):
	print('Turbine ', i, ' yaw angle = ', np.degrees(resPlant.x[i]))
    
# assign yaw angles to turbines
turbines    = [turbine for _, turbine in floris.farm.flow_field.turbine_map.items()]
for i,turbine in enumerate(turbines):
    turbine.yaw_angle = yawOpt[i]
    
# compute the new wake with yaw angles
floris.farm.flow_field.calculate_wake()

# optimal power 
powerOpt = np.sum([turbine.power for turbine in turbines]) 

# plot results
floris.farm.flow_field.plot_flow_field_Zplane()

print('Power increased by ', 100*(powerOpt-power0)/power0)



