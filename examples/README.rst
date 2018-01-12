
FLORIS
------

The inputs to the Floirs model are provided in floris.json and use the NREL 5MW turbine as a reference.  The model parameters used have been published in previous work.  All inputs to the FLORIS model can be changed in the floris.json file.  Note that changing these parameters may result in an unphysical solution.  For questions regarding FLORIS, please contact `Jen Annoni <mailto:jennifer.annoni@nrel.gov>`_, `Paul Fleming <mailto:paul.fleming@nrel.gov>`_, or `Rafael Mudafort <mailto:rafael.mudafort@nrel.gov>`_.


example_script.py
=================

This script provides an example of how to execute Floris.  In particular, this script:

	1. Loads the input file "floris.json" and initializes Floris
		``floris = Floris("floris.json")``

	2. Computes the local power coefficient, thrust coefficients, power, axial induction, and wind speeds at each turbine.
		``for coord, turbine in floris.farm.turbine_map.items():
		    print(str(coord) + ":")
		    print("\tCp -", turbine.Cp)
		    print("\tCt -", turbine.Ct)
		    print("\tpower -", turbine.power)
		    print("\tai -", turbine.aI)
		    print("\taverage velocity -", turbine.get_average_velocity())``

	3. Plot the flow field at a horizontal slice.  In this example, the flow field is plotted at 50% (0.5) of the total z domain, which is 2x the hub height.  
		``floris.farm.flow_field.plot_z_planes([0.2, 0.5, 0.8])``


example_optimization.py
=======================



There is absolutely no warranty on the optimization results.  


FLORIS_Run_Notebook
===================


Future work
===========
Coming soon

License
=======

Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
