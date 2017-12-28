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

