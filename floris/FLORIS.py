"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

# FLORIS driver program

import sys
from InputReader import InputReader

class FLORIS():
    def __init__(self):
        self.input_reader = InputReader()
    
    def process_input(self, input_file):
        self.farm = self.input_reader.input_reader(input_file)

if __name__=="__main__":
    floris = FLORIS()
    floris.process_input(sys.argv[1])
    # output handling
    for coord in floris.farm.get_turbine_coords():
        turbine = floris.farm.get_turbine_at_coord(coord)
        print(str(coord) + ":")
        print("\tCp -", turbine.Cp)
        print("\tCt -", turbine.Ct)
        print("\tpower -", turbine.power)
        print("\tai -", turbine.aI)
        print("\taverage velocity -", turbine.get_average_velocity())
    floris.farm.flow_field.plot_flow_field_planes([0.2])
