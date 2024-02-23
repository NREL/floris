
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from floris.tools import FlorisInterface


"""
For each turbine in the turbine library, make a small figure showing that its power
curve and power loss to yaw are reasonable and  reasonably smooth
"""
ws_array = np.arange(0.1,30,0.2)
wd_array = 270.0 * np.ones_like(ws_array)
yaw_angles = np.linspace(-30,30,60)
wind_speed_to_test_yaw = 11

# Grab the gch model
fi = FlorisInterface("inputs/gch.yaml")

# Make one turbine sim
fi.reinitialize(layout_x=[0], layout_y=[0])

# Apply wind speeds
fi.reinitialize(wind_speeds=ws_array, wind_directions=wd_array)

# Get a list of available turbine models provided through FLORIS, and remove
# multi-dimensional Cp/Ct turbine definitions as they require different handling
turbines = [
    t.stem
    for t in fi.floris.farm.internal_turbine_library.iterdir()
    if t.suffix == ".yaml" and ("multi_dim" not in t.stem)
]

# Declare a set of figures for comparing cp and ct across models
fig_pow_ct, axarr_pow_ct = plt.subplots(2,1,sharex=True,figsize=(10,10))

# For each turbine model available plot the basic info
for t in turbines:

    # Set t as the turbine
    fi.reinitialize(turbine_type=[t])

    # Since we are changing the turbine type, make a matching change to the reference wind height
    fi.assign_hub_height_to_ref_height()

    # Plot power and ct onto the fig_pow_ct plot
    axarr_pow_ct[0].plot(
        fi.floris.farm.turbine_map[0].power_thrust_table["wind_speed"],
        fi.floris.farm.turbine_map[0].power_thrust_table["power"],label=t
    )
    axarr_pow_ct[0].grid(True)
    axarr_pow_ct[0].legend()
    axarr_pow_ct[0].set_ylabel('Power (kW)')
    axarr_pow_ct[1].plot(
        fi.floris.farm.turbine_map[0].power_thrust_table["wind_speed"],
        fi.floris.farm.turbine_map[0].power_thrust_table["thrust_coefficient"],label=t
    )
    axarr_pow_ct[1].grid(True)
    axarr_pow_ct[1].legend()
    axarr_pow_ct[1].set_ylabel('Ct (-)')
    axarr_pow_ct[1].set_xlabel('Wind Speed (m/s)')

    # Create a figure
    fig, axarr = plt.subplots(1,2,figsize=(10,5))

    # Try a few density
    for density in [1.15,1.225,1.3]:

        fi.reinitialize(air_density=density)

        # POWER CURVE
        ax = axarr[0]
        fi.reinitialize(wind_speeds=ws_array, wind_directions=wd_array)
        fi.calculate_wake()
        turbine_powers = fi.get_turbine_powers().flatten() / 1e3
        if density == 1.225:
            ax.plot(ws_array,turbine_powers,label='Air Density = %.3f' % density, lw=2, color='k')
        else:
            ax.plot(ws_array,turbine_powers,label='Air Density = %.3f' % density, lw=1)
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Wind Speed (m/s)')
        ax.set_ylabel('Power (kW)')

        # Power loss to yaw, try a range of yaw angles
        ax = axarr[1]

        fi.reinitialize(wind_speeds=[wind_speed_to_test_yaw], wind_directions=[270.0])
        yaw_result = []
        for yaw in yaw_angles:
            fi.calculate_wake(yaw_angles=np.array([[yaw]]))
            turbine_powers = fi.get_turbine_powers().flatten() / 1e3
            yaw_result.append(turbine_powers[0])
        if density == 1.225:
            ax.plot(yaw_angles,yaw_result,label='Air Density = %.3f' % density, lw=2, color='k')
        else:
            ax.plot(yaw_angles,yaw_result,label='Air Density = %.3f' % density, lw=1)
        # ax.plot(yaw_angles,yaw_result,label='Air Density = %.3f' % density)
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Yaw Error (deg)')
        ax.set_ylabel('Power (kW)')
        ax.set_title('Wind Speed = %.1f' % wind_speed_to_test_yaw )

    # Give a suptitle
    fig.suptitle(t)

plt.show()
