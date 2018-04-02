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
import numpy as np
from scipy.optimize import minimize
import warnings

warnings.simplefilter('ignore', RuntimeWarning)

# optimize wake steering for power maximization
def optimize_plant(x,floris):    
    
    # assign yaw angles to turbines
    turbines    = [turbine for _, turbine in floris.farm.flow_field.turbine_map.items()]
    for i,turbine in enumerate(turbines):
        turbine.yaw_angle = x[i]
     
    floris.farm.flow_field.calculate_wake()
   
    turbines    = [turbine for _, turbine in floris.farm.flow_field.turbine_map.items()]
    power       = -np.sum([turbine.power for turbine in turbines]) 

    return power/(10**3)

def wake_steering(floris,minimum_yaw_angle,maximum_yaw_angle):

	# set initial conditions
	x0 = []
	bnds = []

	turbines    = [turbine for _, turbine in floris.farm.flow_field.turbine_map.items()]
	x0          = [turbine.yaw_angle for turbine in turbines]
	bnds        = [(np.radians(minimum_yaw_angle), np.radians(maximum_yaw_angle)) for turbine in turbines]
	power0      = np.sum([turbine.power for turbine in turbines]) 

	print('=====================================================================')
	print('Optimizing wake redirection control...')
	print('Number of parameters to optimize = ', len(x0))
	print('=====================================================================')

	residual_plant = minimize(optimize_plant,x0,args=(floris),method='SLSQP',bounds=bnds,options={'eps':np.radians(5.0)})

	if np.sum(residual_plant.x) == 0:
		print('No change in controls suggested for this inflow condition...')

	# %%
	opt_yaw_angles = residual_plant.x

	return opt_yaw_angles


