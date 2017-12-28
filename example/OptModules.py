# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 17:27:19 2017

@author: jannoni
"""

import numpy as np

# optimize wake steering for power maximization
def optPlant(x,floris):    
    
    # assign yaw angles to turbines
    turbines    = [turbine for _, turbine in floris.farm.flow_field.turbine_map.items()]
    for i,turbine in enumerate(turbines):
        turbine.yaw_angle = x[i]
     
    floris.farm.flow_field.calculate_wake()
   
    turbines    = [turbine for _, turbine in floris.farm.flow_field.turbine_map.items()]
    power       = -np.sum([turbine.power for turbine in turbines]) 

    return power

