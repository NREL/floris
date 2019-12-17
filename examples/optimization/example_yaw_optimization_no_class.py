import pyoptsparse

import floris.tools as wfct

# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface('../example_input.json')

def objective_function(varDict, **kwargs):
    # Parse the variable dictionary
    yaw = varDict['yaw']

    # Compute the objective function
    funcs = {}
    funcs['obj'] = -1*fi.get_farm_power_for_yaw_angle(yaw)/1e5

    fail = False
    return funcs, fail

# Setup the optimization problem
optProb = pyoptsparse.Optimization('yaw_opt', objective_function)

# Add the design variables to the optimization problem
optProb.addVarGroup('yaw', 4, 'c', lower=0, upper= 20, value=2.)

# Add the objective to the optimization problem
optProb.addObj('obj')

# Setup the optimization solver
# Note: pyOptSparse has other solvers available; some may require additional
#   licenses/installation. See https://github.com/mdolab/pyoptsparse for more
#   information. When ready, they can be invoked by changing 'SLSQP' to the
#   solver name, for example: 'opt = pyoptsparse.SNOPT(fi=fi)'.
opt = pyoptsparse.SLSQP(fi=fi)

# Run the optimization with finite-differencing
solution = opt(optProb, sens='FD')
print(solution)