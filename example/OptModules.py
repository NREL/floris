# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 17:27:19 2017

@author: jannoni
"""

# optimize wake steering for power maximization
def optPlant(x,floris):

    for i,coord in enumerate(floris.farm.get_turbine_coords()):
        turbine = floris.farm.get_turbine_at_coord(coord)
        turbine.yaw_angle = x[i]
     
    floris.farm.flow_field.calculate_wake()

    power = 0.0
    for i,coord in enumerate(floris.farm.get_turbine_coords()):
        turbine = floris.farm.get_turbine_at_coord(coord)
        power = turbine.power + power
        
    power = -power

    #print(x,power)

    return power

