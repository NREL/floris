"""Example of de-rating for load

TBD.
"""

import matplotlib.pyplot as plt
import numpy as np

import floris.layout_visualization as layoutviz
from floris import FlorisModel
from floris.flow_visualization import visualize_cut_plane


fmodel = FlorisModel("../inputs/gch.yaml")

# Change to the simple-derating model turbine
# (Note this could also be done with the mixed model)
fmodel.set_operation_model("simple-derating")

# Convert to a simple two turbine layout with derating turbines
fmodel.set(layout_x=[0, 126.0 * 7], layout_y=[0.0, 0.0])


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
power_setpoints_front = np.linspace(nom_pow,nom_pow *.7,N)
full_rating = np.ones_like(power_setpoints_front) * 5E6

# Only derate the front turbine
power_setpoints = np.column_stack([power_setpoints_front, full_rating])
fmodel.set(power_setpoints=power_setpoints)
fmodel.run()
turbine_powers_derated = fmodel.get_turbine_powers()

# Compute the mean power
power_sum = np.sum(turbine_powers_derated,axis=1)

# Define de-rating as percent reduction
de_rating = 100 * (nom_pow - power_setpoints_front) / nom_pow

# Grab the load heuristic
voc = fmodel._get_turbine_voc()

# Compute mean load
voc_sum = np.sum(voc,axis=1)

# Plot the results
fig, axarr = plt.subplots(2,2,sharex=True, sharey='row',figsize=(12,8))

ax = axarr[0,0]
ax.plot(de_rating, turbine_powers_derated[:,0]/turbine_powers_derated[0,0],'k')
ax.set_title('Upstream Turbine')
ax.set_ylabel('Power (ratio to upstream full power)')

# Add a rounded green text box to the subplot stating energy lost to derating
textstr = 'Energy lost to derating'
props = {'boxstyle':'round', 'facecolor':'r', 'alpha':0.5}
ax.text(0.05, 0.15, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='bottom', bbox=props)


ax = axarr[0,1]
ax.plot(de_rating, turbine_powers_derated[:,1]/turbine_powers_derated[0,0],'k')
ax.set_title('Downstream Turbine')
textstr = 'Energy gained to reduced waking'
props = {'boxstyle':'round', 'facecolor':'g', 'alpha':0.5}
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


ax = axarr[1,0]
ax.plot(de_rating, voc[:,0]/voc[0,0],'k')
ax.set_ylabel('VOC (ratio to upstream full power)')
ax.set_xlabel('Front De-Rating %')
textstr = 'Lower VOC via lower thrust'
props = {'boxstyle':'round', 'facecolor':'g', 'alpha':0.5}
ax.text(0.15, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


ax = axarr[1,1]
ax.plot(de_rating, voc[:,1]/voc[0,0],'k')
ax.set_xlabel('Front De-Rating %')
textstr = 'Lower VOC via lower waking'
props = {'boxstyle':'round', 'facecolor':'g', 'alpha':0.5}
ax.text(0.15, 0.15, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='bottom', bbox=props)


for ax in axarr.flatten():
    ax.grid(True)

# Save the figure
fig.savefig('derating_grid.png', bbox_inches='tight', dpi=300)

# Calculate the revenue and cost
# Force cost to be equal to 1.1* power at 1.0 energy value
cost = 0.95 * voc_sum * (power_sum[0] / voc_sum[0])

fig, ax = plt.subplots()
for elecpric in [2.0, 1.0, 0.5]:
    revenue = elecpric * power_sum
    profit = revenue - cost
    ax.plot(de_rating, profit-profit[0],label=f'Electricity Price {elecpric}')
ax.set_xlabel('Front De-Rating %')
ax.set_ylabel('Profit (ratio to not derated)')
ax.axhline(0,color='k',linestyle='--')
ax.grid()
# Remove numeric labels on yaxis
ax.set_yticklabels([])
ax.legend()
fig.savefig('elec.png', bbox_inches='tight', dpi=300)






# Visualize the farm
fig, ax = plt.subplots()
horizontal_plane = fmodel.calculate_horizontal_plane(height=90.0)
visualize_cut_plane(
    horizontal_plane,
    ax=ax,
    min_speed=3,
    max_speed=9,
)
layoutviz.plot_waking_directions(fmodel, ax=ax)
layoutviz.plot_turbine_rotors(fmodel, ax=ax)
fig.savefig('layout.png', bbox_inches='tight', dpi=300)

plt.show()
