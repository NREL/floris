"""Example of de-rating for load

TBD.
"""

import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel


fmodel = FlorisModel("../inputs/gch.yaml")

# Change to the simple-derating model turbine
# (Note this could also be done with the mixed model)
fmodel.set_operation_model("simple-derating")

# Convert to a simple two turbine layout with derating turbines
fmodel.set(layout_x=[0, 126.0 * 5], layout_y=[0.0, 0.0])


# Set the wind directions and speeds to be constant over n_findex = N time steps
N = 50
fmodel.set(
    wind_directions=270 * np.ones(N),
    wind_speeds=7.0 * np.ones(N),
    turbulence_intensities=0.06 * np.ones(N),
)
fmodel.run()
turbine_powers_orig = fmodel.get_turbine_powers()

# Get the nominal operating power of the upstream turbine
nom_pow = turbine_powers_orig[0,0]

# # Derate the front turbine from full power to 75 %
power_setpoints_front = np.linspace(nom_pow,nom_pow *.75,N)
full_rating = np.ones_like(power_setpoints_front) * 5E6

# Only derate the front turbine
power_setpoints = np.column_stack([power_setpoints_front, full_rating])
fmodel.set(power_setpoints=power_setpoints)
fmodel.run()
turbine_powers_derated = fmodel.get_turbine_powers()

# Compute the mean power
power_mean = np.mean(turbine_powers_derated,axis=1)

# Define de-rating as percent reduction
de_rating = 100 * (nom_pow - power_setpoints_front) / nom_pow

# Grab the load heuristic
load_h = fmodel._get_turbine_load_h()

# Compute mean load
load_h_mean = np.mean(load_h,axis=1)

# Plot the results
fig, axarr = plt.subplots(2,3,sharex=True, sharey='row',figsize=(12,8))

ax = axarr[0,0]
ax.plot(de_rating, turbine_powers_derated[:,0]/turbine_powers_derated[0,0],'k')
ax.set_title('Turbine 0')
ax.set_ylabel('Power (ratio to nomimal)')

ax = axarr[0,1]
ax.plot(de_rating, turbine_powers_derated[:,1]/turbine_powers_derated[0,1],'k')
ax.set_title('Turbine 1')

ax = axarr[0,2]
ax.plot(de_rating, power_mean/power_mean[0],'k')
ax.set_title('Mean')

ax = axarr[1,0]
ax.plot(de_rating, load_h[:,0]/load_h[0,0],'k')
ax.set_ylabel('Load H (ratio to nomimal)')
ax.set_xlabel('Front De-Rating %')

ax = axarr[1,1]
ax.plot(de_rating, load_h[:,1]/load_h[0,1],'k')
ax.set_xlabel('Front De-Rating %')

ax = axarr[1,2]
ax.plot(de_rating, load_h_mean/load_h_mean[0],'k')
ax.set_xlabel('Front De-Rating %')

for ax in axarr.flatten():
    ax.grid(True)


plt.show()
