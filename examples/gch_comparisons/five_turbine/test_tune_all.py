# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# See read the https://floris.readthedocs.io for documentation


import matplotlib

matplotlib.use('tkagg')
import floris.tools as wfct
import numpy as np

print('Running FLORIS with no yaw...')
# Instantiate the FLORIS object

N = 10
ti_initial = np.linspace(0.1, 0.9, N)
ti_constant = np.linspace(0.1, 0.9, N)
ti_ai = np.linspace(0.1, 0.9, N)
ti_downstream = np.linspace(-0.9, -0.1, N)

minErr = 1000000000
opt_params = np.zeros(4)

fi = wfct.floris_interface.FlorisInterface("../../example_input.json")

fi.floris.farm.wake._velocity_model.use_yaw_rec = True
fi.floris.farm.wake._deflection_model.use_yaw_eff = True

# Set turbine locations to 3 turbines in a row
D = fi.floris.farm.turbines[0].rotor_diameter

l_x = [0, 6 * D, 12 * D, 18 * D, 24 * D]
# l_x = [0,7*D,14*D]
l_y = [0, 0, 0, 0, 0]
count = 0
fi.reinitialize_flow_field(layout_array=(l_x, l_y), wind_direction=270)
for i in range(N):
    for j in range(N):
        for k in range(N):
            for l in range(N):
                print(count, 'out of ', N ** 4)
                count = count + 1
                fi.floris.farm.flow_field.wake.velocity_model.ti_initial = ti_initial[i]
                fi.floris.farm.flow_field.wake.velocity_model.ti_constant = ti_constant[j]
                fi.floris.farm.flow_field.wake.velocity_model.ti_ai = ti_ai[k]
                fi.floris.farm.flow_field.wake.velocity_model.ti_downstream = ti_downstream[l]

                # fi.reinitialize_flow_field(layout_array=(layout_x, layout_y),wind_direction=wind_direction)
                fi.reinitialize_flow_field(turbulence_intensity=0.09)
                fi.calculate_wake(yaw_angles=np.zeros(len(l_x)))

                # Initial power output
                power_initial = fi.get_farm_power()
                SOWFA_base_hi = 1000 * np.array([843.9, 856.9, 893.1, 926.2])
                GCH_base_hi = fi.get_turbine_power()[1:]
                # print('Initial farm power = ', power_initial)

                # # =============================================================================
                # print('Finding optimal yaw angles in FLORIS...')
                # # =============================================================================

                # Set bounds for allowable wake steering
                min_yaw = 0.0
                max_yaw = 25.0
                yaw_angles = [25, 25, 25, 0, 0]

                fi.reinitialize_flow_field()
                # print('==========================================')
                fi.calculate_wake(yaw_angles=yaw_angles)
                power_opt = fi.get_farm_power()
                SOWFA_opt_hi = 1000 * np.array([988.4, 1030.0, 1443.8, 1141.8])
                GCH_opt_hi = fi.get_turbine_power()[1:]
                gain_hi = 100. * (power_opt - power_initial) / power_initial
                print('==========================================')
                print(ti_initial[i], ti_constant[j], ti_ai[k], ti_downstream[l])
                print('Total Power Gain HI TI = %.1f%%' %
                      (100. * (power_opt - power_initial) / power_initial))
                # print('==========================================')

                # # =============================================================================
                # print('Finding optimal yaw angles in FLORIS low ti...')
                # # =============================================================================

                # Set bounds for allowable wake steering
                min_yaw = 0.0
                max_yaw = 25.0

                # Instantiate the Optimization object
                fi.reinitialize_flow_field(turbulence_intensity=0.065)
                fi.calculate_wake(yaw_angles=np.zeros(len(l_x)))

                # Initial power output
                power_initial = fi.get_farm_power()
                SOWFA_base = 1000 * np.array([654.7, 764.8, 825., 819.8])
                GCH_base = fi.get_turbine_power()[1:]

                # Perform optimization
                yaw_angles = [25, 25, 25, 0, 0]
                # print('==========================================')
                fi.reinitialize_flow_field()
                fi.calculate_wake(yaw_angles=yaw_angles)
                power_opt = fi.get_farm_power()
                SOWFA_opt = 1000 * np.array([929.8, 1083.5, 1425.5, 1105.1])
                GCH_opt = fi.get_turbine_power()[1:]
                # print('==========================================')
                gain_low = 100. * (power_opt - power_initial) / power_initial
                print('Total Power Gain Low TI = %.1f%%' %
                      (100. * (power_opt - power_initial) / power_initial))
                # print('==========================================')
                err = np.sum(
                    (SOWFA_base - GCH_base) ** 2 + (SOWFA_opt - GCH_opt) ** 2 + (SOWFA_base_hi - GCH_base_hi) ** 2 + (
                                SOWFA_opt_hi - GCH_opt_hi) ** 2) / (10 ** 3)
                print('err = ', err, minErr)
                if err < minErr:
                    minErr = err
                    print('found min error: ', i, j, k, l)
                    opt_params[0] = i
                    opt_params[1] = j
                    opt_params[2] = k
                    opt_params[3] = l

print('Optimal parameters:')
print(opt_params)
print('ti_initial = ', ti_initial[int(opt_params[0])])
print('ti_constant = ', ti_constant[int(opt_params[1])])
print('ti_ai = ', ti_ai[int(opt_params[2])])
print('ti_downstream = ', ti_downstream[int(opt_params[3])])

fi.floris.farm.flow_field.wake.velocity_model.ti_initial = ti_initial[int(opt_params[0])]
fi.floris.farm.flow_field.wake.velocity_model.ti_constant = ti_constant[int(opt_params[1])]
fi.floris.farm.flow_field.wake.velocity_model.ti_ai = ti_ai[int(opt_params[2])]
fi.floris.farm.flow_field.wake.velocity_model.ti_downstream = ti_downstream[int(opt_params[3])]

# HI TI
fi.reinitialize_flow_field(turbulence_intensity=0.09)
fi.calculate_wake(yaw_angles=np.zeros(len(l_x)))
# Initial power output
power_initial = fi.get_farm_power()
# Set bounds for allowable wake steering
yaw_angles = [25, 25, 25, 0, 0]
fi.reinitialize_flow_field()
fi.calculate_wake(yaw_angles=yaw_angles)
power_opt = fi.get_farm_power()
gain_hi = 100. * (power_opt - power_initial) / power_initial
print('==========================================')
print(ti_initial[i], ti_constant[j], ti_ai[k], ti_downstream[l])
print('Total Power Gain HI TI = %.1f%%' %
      (100. * (power_opt - power_initial) / power_initial))
# print('==========================================')

# LOW TI

# Instantiate the Optimization object
fi.reinitialize_flow_field(turbulence_intensity=0.065)
fi.calculate_wake(yaw_angles=np.zeros(len(l_x)))
# Initial power output
power_initial = fi.get_farm_power()
# Perform optimization
yaw_angles = [25, 25, 25, 0, 0]
fi.reinitialize_flow_field()
fi.calculate_wake(yaw_angles=yaw_angles)
power_opt = fi.get_farm_power()
gain_low = 100. * (power_opt - power_initial) / power_initial
print('Total Power Gain Low TI = %.1f%%' %
      (100. * (power_opt - power_initial) / power_initial))




