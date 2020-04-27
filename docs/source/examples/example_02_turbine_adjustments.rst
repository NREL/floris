example_02_turbine_adjustments.py 
================================= 

This example illustrates changing the properties of some of the turbines
This can be used to setup farms of different turbines

.. code-block:: python3 

	import matplotlib.pyplot as plt
	import floris.tools as wfct

Initialize the FLORIS interface fi

.. code-block:: python3 

	fi = wfct.floris_interface.FlorisInterface("../example_input.json")

Set to 2x2 farm

.. code-block:: python3 

	fi.reinitialize_flow_field(layout_array=[[0,0,600,600],[0,300,0,300]])

Change turbine 0 and 3 to have a 35 m rotor diameter

.. code-block:: python3 

	fi.change_turbine([0,3],{'rotor_diameter':35})

Calculate wake

.. code-block:: python3 

	fi.calculate_wake()

Get horizontal plane at default height (hub-height)

.. code-block:: python3 

	hor_plane = fi.get_hor_plane()

Plot and show

.. code-block:: python3 

	fig, ax = plt.subplots()
	wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
	plt.show()
