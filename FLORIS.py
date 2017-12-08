"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

# FLORIS driver program

# import specific models
from turbines.NREL5MW import NREL5MW
from wakes.JensenJimenez import JensenJimenez
from src.models.WakeCombination import WakeCombination
from farms.TwoByTwo import TwoByTwo
from src.io.InputReader import InputReader


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
                    combination=WakeCombination("fls"))

# t0 = twobytwo.getTurbineAtCoord((0,0))
# print("t0.Cp", t0.Cp)
# print("t0.Ct", t0.Ct)
# print("t0.power", t0.power)
# print("t0.aI", t0.aI)
# print("t0.get_average_velocity()", t0.get_average_velocity())

ff = twobytwo.get_flow_field()
ff.plot_flow_field_plane()
