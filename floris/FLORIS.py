"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

# FLORIS driver program

# import specific models
from InputReader import InputReader
from WakeCombination import WakeCombination

# to be deprecated by json io
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wakes.JensenJimenez import JensenJimenez
from farms.TwoByTwo import TwoByTwo


inputReader = InputReader()

# turbine input
turbineInput = "turbines/NREL5MW.json"
turbine = inputReader.buildTurbine(turbineInput)

# wake input
wakeInput = "wakes/JensenJimenez.json"

# farm input
farmInput = "farms/TwoByTwo.json"
# TODO: add controls to farm

twobytwo = TwoByTwo(turbine=turbine,
                    wake=JensenJimenez(),
                    combination=WakeCombination("sosfs"))

# output handling
for coord in twobytwo.get_turbine_coords():
    turbine = twobytwo.get_turbine_at_coord(coord)
    print(str(coord) + ":")
    print("\tCp -", turbine.Cp)
    print("\tCt -", turbine.Ct)
    print("\tpower -", turbine.power)
    print("\tai -", turbine.aI)
    print("\taverage velocity -", turbine.get_average_velocity())

ff = twobytwo.get_flow_field()
ff.plot_flow_field_planes([0.2])#, 0.5, 0.8])
