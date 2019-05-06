example_0015_open_and_vis_sowfa.py
==================================

The code for this example can be found here: 
`example_0015_open_and_vis_sowfa.py 
<https://github.com/NREL/floris/blob/develop/examples/example_0015_open_and_vis_sowfa.py>`_

SOWFA (https://github.com/NREL/SOWFA) is an NREL LES-code which is used to 
explore wind farm controls in high-fidelity and compare FLORIS results to. This 
repository therefore includes some tools for interacting with SOWFA in a way 
that standardizes the interface to FLORIS and SOWFA and allows a common set of 
analysis functions to be applied to the outputs of each.  This first example 
shows a basic example of working with a saved SOWFA run.

Note that an example saved SOWFA run is included with this repository.  It is 
anticipated that as SOWFA (or NALU) evolves, these utilities may well need to 
evolve with them.

The first section of the code imports the saved run by naming the case 
directory and using the print method to summarize the case.

::

    sowfa_case = wfct.sowfa_utilities.SowfaInterface('sowfa_example')

    # Summarize self
    print(sowfa_case)

The second section uses the same visualization applied previously to FLORIS, 
now applied to the saved time-averaged SOWFA flow field.  The flow data was 
generated using OpenFoam output functions which can be observed in 
``examples/sowfa_example/system/controlDict``.

The third case accesses and plots the outputs of the turbines in the flow.  In 
this example there is only one turbine, but the syntax for selecting between 
turbines is still provided to illustrate that the turbines are identified using 
a column within the output dataframe.