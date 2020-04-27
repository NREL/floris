example_03_get_and_set_model_parameters.py 
========================================== 


.. code-block:: python3 

	import matplotlib.pyplot as plt
	import floris.tools as wfct

Initialize the FLORIS interface fi

.. code-block:: python3 

	fi = wfct.floris_interface.FlorisInterface("../example_input.json")

Show the current model parameters

.. code-block:: python3 

	print('All the model parameters and their current values:\n')
	fi.show_model_parameters()
	print('\n')

Show the current model parameters with docstring info

.. code-block:: python3 

	print('All the model parameters, their current values, and docstrings:\n')
	fi.show_model_parameters(verbose=True)
	print('\n')

Show a specific model parameter with its docstring

.. code-block:: python3 

	print('A specific model parameter, its current value, and its docstring:\n')
	fi.show_model_parameters(params=['ka'], verbose=False)
	print('\n')

Get the current model parameters

.. code-block:: python3 

	model_params = fi.get_model_parameters()
	print('The current model parameters:\n')
	print(model_params)
	print('\n')

Set parameters on the current model

.. code-block:: python3 

	print('Set specific model parameters on the current wake model:\n')
	params = {
	    'Wake Velocity Parameters': {'alpha': 0.2},
	    'Wake Deflection Parameters': {'alpha': 0.2},
	    'Wake Turbulence Parameters': {'ti_constant': 1.0}
	}
	fi.set_model_parameters(params)
	print('\n')

Check that the parameters were changed

.. code-block:: python3 

	print('Observe that the requested paremeters changes have been made:\n')
	model_params = fi.get_model_parameters()
	print(model_params)
	print('\n')
