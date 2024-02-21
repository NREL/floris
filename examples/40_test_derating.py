
import matplotlib.pyplot as plt
import numpy as np
import yaml

from floris.tools import FlorisInterface


"""
Example to test out derating of turbines and mixed derating and yawing. Will be refined before
release. TODO: Demonstrate shutting off turbines also, once developed.
"""

# Grab model of FLORIS and update to deratable turbines
fi = FlorisInterface("inputs/gch.yaml")

with open(str(
    fi.floris.as_dict()["farm"]["turbine_library_path"] /
    (fi.floris.as_dict()["farm"]["turbine_type"][0] + ".yaml")
)) as t:
    turbine_type = yaml.safe_load(t)
turbine_type["power_thrust_model"] = "simple-derating"

# Convert to a simple two turbine layout with derating turbines
fi.reinitialize(layout_x=[0, 1000.0], layout_y=[0.0, 0.0], turbine_type=[turbine_type])

# Set the wind directions and speeds to be constant over n_findex = N time steps
N = 50
fi.reinitialize(wind_directions=270 * np.ones(N), wind_speeds=10.0 * np.ones(N))
fi.calculate_wake()
turbine_powers_orig = fi.get_turbine_powers()

# Add derating
power_setpoints = np.tile(np.linspace(1, 6e6, N), 2).reshape(2, N).T
fi.calculate_wake(power_setpoints=power_setpoints)
turbine_powers_derated = fi.get_turbine_powers()

# Compute available power at downstream turbine
power_setpoints_2 = np.array([np.linspace(1, 6e6, N), np.full(N, None)]).T
fi.calculate_wake(power_setpoints=power_setpoints_2)
turbine_powers_avail_ds = fi.get_turbine_powers()[:,1]

# Plot the results
fig, ax = plt.subplots(1, 1)
ax.plot(power_setpoints[:, 0]/1000, turbine_powers_derated[:, 0]/1000, color="C0", label="Upstream")
ax.plot(
    power_setpoints[:, 1]/1000,
    turbine_powers_derated[:, 1]/1000,
    color="C1",
    label="Downstream"
)
ax.plot(
    power_setpoints[:, 0]/1000,
    turbine_powers_orig[:, 0]/1000,
    color="C0",
    linestyle="dotted",
    label="Upstream available"
)
ax.plot(
    power_setpoints[:, 1]/1000,
    turbine_powers_avail_ds/1000,
    color="C1",
    linestyle="dotted", label="Downstream available"
)
ax.plot(
    power_setpoints[:, 1]/1000,
    np.ones(N)*np.max(turbine_type["power_thrust_table"]["power"]),
    color="k",
    linestyle="dashed",
    label="Rated power"
)
ax.grid()
ax.legend()
ax.set_xlim([0, 6e3])
ax.set_xlabel("Power setpoint (kW)")
ax.set_ylabel("Power produced (kW)")

# Second example showing mixed model use.
turbine_type["power_thrust_model"] = "mixed"
yaw_angles = np.array([
    [0.0, 0.0],
    [0.0, 0.0],
    [20.0, 10.0],
    [0.0, 10.0],
    [20.0, 0.0]
])
power_setpoints = np.array([
    [None, None],
    [2e6, 1e6],
    [None, None],
    [2e6, None,],
    [None, 1e6]
])
fi.reinitialize(
    wind_directions=270 * np.ones(len(yaw_angles)),
    wind_speeds=10.0 * np.ones(len(yaw_angles)),
    turbine_type=[turbine_type]*2
)
fi.calculate_wake(yaw_angles=yaw_angles, power_setpoints=power_setpoints)
turbine_powers = fi.get_turbine_powers()
print(turbine_powers)

plt.show()
