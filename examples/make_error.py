

from floris.tools import FlorisInterface


fi = FlorisInterface("inputs/gch.yaml")

# # Convert to a simple two turbine layout
# fi.reinitialize( layout=( [0, 500.], [0., 0.] ) )


# fi.calculate_wake()

# Get the turbine powers
turbine_powers = fi.get_turbine_powers()/1000.
