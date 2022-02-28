# Copyright 2022 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

from time import perf_counter as timerpc
import numpy as np
import matplotlib.pyplot as plt
from floris.tools import FlorisInterface
from floris.tools.optimization.yaw_optimization.yaw_optimizer_scipy import (
    YawOptimizationScipy
)
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import (
    YawOptimizationSR
)


"""
This example compares the SciPy-based yaw optimizer with the new Serial-Refine optimizer.

First, we initialize our Floris Interface, and then generate a 3 turbine wind farm. Next, we create two yaw optimization objects, `yaw_opt_sr` and `yaw_opt_scipy` for the Serial-Refine and SciPy methods, respectively. We then perform the optimization using both methods. Finally, we compare the time it took to find the optimal angles and plot the optimal yaw angles and resulting wind farm powers.
"""

# Load the default example floris object
fi = FlorisInterface("inputs/gch.yaml") # GCH model matched to the default "legacy_gauss" of V2
# fi = FlorisInterface("inputs/cc.yaml") # New CumulativeCurl model

# Reinitialize as a 3-turbine farm with range of WDs and 1 WS
D = 126.0 # Rotor diameter for the NREL 5 MW
fi.reinitialize(
    layout=[[0.0, 5 * D, 10 * D], [0.0, 0.0, 0.0]],
    wind_directions=np.arange(0.0, 360.0, 3.0), 
    wind_speeds=[8.0],
)

print("Performing optimizations with SciPy...")
start_time = timerpc()
yaw_opt_scipy = YawOptimizationScipy(fi)
df_opt_scipy = yaw_opt_scipy.optimize()
time_scipy = timerpc() - start_time

print("Performing optimizations with Serial Refine...")
start_time = timerpc()
yaw_opt_sr = YawOptimizationSR(fi)
df_opt_sr = yaw_opt_sr.optimize()
time_sr = timerpc() - start_time

# Print time spent
print("\n Time spent, Serial Refine: {:.2f} s.".format(time_sr))
print(" Time spent, SciPy (SLSQP): {:.2f} s.\n".format(time_scipy))

# Split out the turbine results
yaw_angles_opt_sr = np.vstack(df_opt_sr.yaw_angles_opt)
yaw_angles_opt_scipy = np.vstack(df_opt_scipy.yaw_angles_opt)

# Yaw results
for t in range(3):
    fig, ax = plt.subplots()
    ax.plot(df_opt_sr.wind_direction, yaw_angles_opt_sr[:, t],color='r',label='Serial Refine')
    ax.plot(df_opt_scipy.wind_direction, yaw_angles_opt_scipy[:, t],'--', color='g', label='SciPy')
    ax.grid(True)
    ax.set_ylabel('Yaw Offset (deg')
    ax.legend()
    ax.grid(True)
    ax.set_title("Turbine {:d}".format(t))

# Power results
fig, ax = plt.subplots()
ax.plot(df_opt_sr.wind_direction,df_opt_sr.farm_power_baseline,color='k',label='Baseline')
ax.plot(df_opt_sr.wind_direction,df_opt_sr.farm_power_opt,color='r',label='Optimized, Serial Refine')
ax.plot(df_opt_scipy.wind_direction,df_opt_scipy.farm_power_opt,'--', color='g', label='Optimized, SciPy')
ax.set_ylabel('Wind Farm Power (W)')
ax.set_xlabel('Wind Direction (deg)')
ax.legend()
ax.grid(True)

plt.show()
