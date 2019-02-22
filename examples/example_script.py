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
from floris import Floris
from copy import deepcopy
from floris.visualization import VisualizationManager

if len(sys.argv) > 1:
    floris = Floris(sys.argv[1])
else:
    floris = Floris("example_input.json")
floris.farm.flow_field.calculate_wake()

for coord, turbine in floris.farm.turbine_map.items():
    print(str(coord) + ":")
    print("\tCp -", turbine.Cp)
    print("\tCt -", turbine.Ct)
    print("\tpower -", turbine.power)
    print("\tai -", turbine.aI)
    print("\taverage velocity -", turbine.average_velocity)

# Visualization
ff_viz = deepcopy(floris.farm.flow_field)
visualization_grid_resolution = (100, 100, 25)
visualization_manager = VisualizationManager(ff_viz, visualization_grid_resolution)
visualization_manager.plot_z_planes([1/6.])
visualization_manager.plot_x_planes([0.5])
visualization_manager.show()
