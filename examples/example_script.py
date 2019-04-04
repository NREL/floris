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
from floris.simulation import Floris
import copy

# Initialize the floris object with a json input file
if len(sys.argv) > 1:
    floris = Floris(sys.argv[1])
else:
    floris = Floris("example_input.json")

# Setup the Gauss velocity and deflection models
gauss_floris = copy.deepcopy(floris)
gauss_floris.farm.set_wake_model("gauss", calculate_wake=True)

# Display the results
print("Gauss model results")
print("{:>30} | {:<7} {:<7} {:<15} {:<15} {:<15}".format("turbine location", "Cp", "Ct", "Power", "AxialInduction", "AverageVelocity"))
gauss_items = gauss_floris.farm.turbine_map.items
for coord, turbine in gauss_items:
    print("{} | {:<7.3f} {:<7.3f} {:<15.3f} {:<15.3f} {:<15.3f}".format(coord, turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity))
print("")

# Setup the Curl velocity and deflection models
curl_floris = copy.deepcopy(floris)
curl_floris.farm.set_wake_model("curl", calculate_wake=True)

# Display the results
print("Curl model results")
print("{:>30} | {:<7} {:<7} {:<15} {:<15} {:<15}".format("turbine location", "Cp", "Ct", "Power", "AxialInduction", "AverageVelocity"))
curl_items = curl_floris.farm.turbine_map.items
for coord, turbine in curl_items:
    print("{} | {:<7.3f} {:<7.3f} {:<15.3f} {:<15.3f} {:<15.3f}".format(coord, turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity))
print("")

# calculate and display the difference
print("Difference (absolute difference)")
print("{:>30} | {:<7} {:<7} {:<15} {:<15} {:<15}".format("turbine location", "Cp", "Ct", "Power", "AxialInduction", "AverageVelocity"))
for gauss, curl in zip(gauss_items, curl_items):
    gauss_coord, gauss_turbine = gauss[0], gauss[1]
    curl_coord, curl_turbine = curl[0], curl[1]
    print("{} | {:<7.3f} {:<7.3f} {:<15.3f} {:<15.3f} {:<15.3f}".format(
        gauss_coord,
        abs(gauss_turbine.Cp - curl_turbine.Cp),
        abs(gauss_turbine.Ct - curl_turbine.Ct),
        abs(gauss_turbine.power - curl_turbine.power),
        abs(gauss_turbine.aI - curl_turbine.aI),
        abs(gauss_turbine.average_velocity - curl_turbine.average_velocity))
    )
