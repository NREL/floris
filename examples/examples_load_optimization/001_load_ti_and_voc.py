"""Example: Load turbulence intensity

Demonstrate the behavior of the load turbulence intensity model and
variable operating cost (VOC) model with respect
to changing wind direction (which changes wake interactions) and
changing power setpoints (which changes wake strength).

"""

import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel, TimeSeries
from floris.core.turbine.operation_models import (
    POWER_SETPOINT_DEFAULT,
    POWER_SETPOINT_DISABLED,
)
from floris.optimization.load_optimization.load_optimization import (
    compute_load_ti,
    compute_turbine_voc,
)


# Declare a floris model with default configuration
fmodel = FlorisModel(configuration="defaults")
fmodel.set_operation_model("simple-derating")

## Wind direction sweep
wind_directions = np.arange(0, 360, 1.0)

# Assume uniform load ambient turbulence intensities
load_ambient_tis = 0.1 * np.ones_like(wind_directions)

# Declare a time series representing the wind direction sweep
time_series = TimeSeries(
    wind_directions=wind_directions, wind_speeds=8.0, turbulence_intensities=0.06
)

# Set the turbine layout to be a simple two turbine layout using the
# time series object
D = fmodel.core.farm.rotor_diameters[0]
fmodel.set(layout_x=[0, D * 7], layout_y=[0.0, 0.0], wind_data=time_series)
fmodel.run()

# Compute the load turbulence intensity and VOC
load_ti = compute_load_ti(fmodel, load_ambient_tis)
voc = compute_turbine_voc(fmodel, A=0.01, load_ambient_tis=load_ambient_tis)

# Plot the TI and VOC for each turbine
fig, ax = plt.subplots(2, 1, sharex=True)
for t in range(fmodel.n_turbines):
    ax[0].plot(wind_directions, load_ti[:, t], label=f"Turbine {t}")
    ax[1].plot(wind_directions, voc[:, t], label=f"Turbine {t}")
ax[0].set_ylabel("Load TI [-]")
ax[1].set_ylabel("VOC [$]")
ax[1].set_xlabel("Wind Direction [deg]")
ax[1].legend(loc="lower right")
ax[0].grid()
ax[1].grid()
ax[0].set_title("Load TI and VOC vs Wind Direction")

## Power setpoint sweep
N = 50
time_series = TimeSeries(
    wind_directions=np.ones(N) * 270.0, # Single wind direction
    wind_speeds=8.0,
    turbulence_intensities=0.06,
)
power_setpoints = np.column_stack([
    np.linspace(5e6, POWER_SETPOINT_DISABLED, N),
    np.ones(N)*POWER_SETPOINT_DEFAULT
])
load_ambient_tis = np.ones(N) * 0.1
fmodel.set(
    layout_x=[0, D * 7],
    layout_y=[0.0, 0.0],
    wind_data=time_series,
    power_setpoints=power_setpoints,
)
fmodel.run()

# Compute the load turbulence intensity and VOC
load_ti = compute_load_ti(fmodel, load_ambient_tis)
voc = compute_turbine_voc(fmodel, A=0.01, load_ambient_tis=load_ambient_tis)

# Plot the TI and VOC for each turbine
fig, ax = plt.subplots(2, 1, sharex=True)
for t in range(fmodel.n_turbines):
    ax[0].plot(power_setpoints[:,0], load_ti[:, t], label=f"Turbine {t}")
    ax[1].plot(power_setpoints[:,0], voc[:, t], label=f"Turbine {t}")
ax[0].set_ylabel("Load TI [-]")
ax[1].set_ylabel("VOC [$]")
ax[1].set_xlabel("Power Setpoint (T0) [W]")
ax[1].legend(loc="lower right")
ax[0].grid()
ax[1].grid()
ax[0].set_title("Load TI and VOC vs T0 Power Setpoint")

plt.show()
