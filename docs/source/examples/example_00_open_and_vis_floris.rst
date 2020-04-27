example_00_open_and_vis_floris.py 
================================= 


.. code-block:: python3 

	import matplotlib.pyplot as plt
	import floris.tools as wfct

Initialize the FLORIS interface fi
For basic usage, the florice interface provides a simplified interface to
the underlying classes

.. code-block:: python3 

	fi = wfct.floris_interface.FlorisInterface("../example_input.json")

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
