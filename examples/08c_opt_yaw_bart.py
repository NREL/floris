import numpy as np
import matplotlib.pyplot as plt
from floris.tools import FlorisInterface, UncertaintyInterface
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import (
    YawOptimizationSR,
)

# Load the default example floris object
fi = FlorisInterface("inputs/gch.yaml") # GCH model matched to the default "legacy_gauss" of V2
D = 126.0 # Rotor diameter for the NREL 5 MW
fi.reinitialize(
    layout=[[0.0, 5 * D, 10 * D], [0.0, 0.0, 0.0]],
    wind_directions=np.arange(0.0, 360.0, 3.0), 
    wind_speeds=[8.0],
    turbulence_intensity=0.06,
)

# Initialize uncertainty FLORIS object as copy of nominal object
fi_unc = UncertaintyInterface(fi)  

# Initialize optimizer object and run deterministic optimization
yaw_opt = YawOptimizationSR(
    fi,
    minimum_yaw_angle=-25.0,
    maximum_yaw_angle=25.0,
)
df_opt = yaw_opt.optimize()

# Initialize optimizer object and run stochastic optimization
yaw_opt_unc = YawOptimizationSR(
    fi_unc,
    minimum_yaw_angle=-25.0,
    maximum_yaw_angle=25.0,
)
df_opt_unc = yaw_opt_unc.optimize()

# Split out the turbine results
yaw_angles_opt_deterministic = np.vstack(df_opt["yaw_angles_opt"])
yaw_angles_opt_stochastic = np.vstack(df_opt_unc["yaw_angles_opt"])

# Show the results
fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8,8))

# Yaw results
for ti in range(3):
    ax[ti].plot(
        df_opt["wind_direction"],
        yaw_angles_opt_deterministic[:, ti],
        label="Deterministic",
    )
    ax[ti].plot(
        df_opt_unc["wind_direction"],
        yaw_angles_opt_stochastic[:, ti],
        label="Stochastic"
    )
    ax[ti].set_ylabel("Optimal yaw \n angle T{:d} (deg)".format(ti))
    ax[ti].grid(True)
    ax[ti].legend()

# Power results
ax[3].plot(
    df_opt["wind_direction"],
    df_opt["farm_power_baseline"],
    "-",
    color="tab:blue",
    label="Baseline Farm Power (deterministic)"
)
ax[3].plot(
    df_opt["wind_direction"],
    df_opt["farm_power_opt"],
    "--",
    color="tab:blue",
    label="Optimized Farm Power (deterministic)"
)
ax[3].plot(
    df_opt_unc["wind_direction"],
    df_opt_unc["farm_power_baseline"],
    "-",
    color="tab:orange",
    label="Baseline Farm Power (stochastic)"
)
ax[3].plot(
    df_opt_unc["wind_direction"],
    df_opt_unc["farm_power_opt"],
    "--",
    color="tab:orange",
    label="Optimized Farm Power (stochastic)"
)
ax[3].set_ylabel("Power (W)")
ax[3].set_xlabel("Wind Direction (deg)")
ax[3].legend()
ax[3].grid(True)

plt.show()
