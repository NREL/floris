# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


from cProfile import label
import matplotlib.pyplot as plt
import numpy as np

from floris.tools import FlorisInterface

"""
This example creates a FLORIS instance
1) Makes a two-turbine layout
2) Demonstrates single ws/wd simulations
3) Demonstrates mulitple ws/wd simulations

Main concept is introduce FLORIS and illustrate essential structure of most-used FLORIS calls
"""

# Initialize FLORIS with the given input file via FlorisInterface.
# For basic usage, FlorisInterface provides a simplified and expressive
# entry point to the simulation routines.
fi = FlorisInterface("C:\\Users\\gstarke\\Documents\\Research_Programs\\FLORIS\\floris\\examples\\inputs\\gch.yaml")

# Convert to a simple two turbine layout
fi.reinitialize( layout=( [0, 500.], [0., 0.] ) )

# Single wind speed and wind direction
print('\n============================= Single Wind Direction and Wind Speed =============================')

# Get the turbine powers assuming 1 wind speed and 1 wind direction
fi.reinitialize(wind_directions=[270.], wind_speeds=[8.0])

# Set the yaw angles to 0
yaw_angles = np.zeros([1,1,2]) # 1 wind direction, 1 wind speed, 2 turbines
# yaw_angles = np.zeros([2]) # 1 wind direction, 1 wind speed, 2 turbines
CT_input = np.zeros([1,1,2])+0.75 # 1 wind direction, 1 wind speed, 2 turbines
fi.calculate_wake(yaw_angles=yaw_angles, CT_inputs=CT_input)
print('did it get here?')


# Get the turbine powers
turbine_powers = fi.get_turbine_powers()/1000.
print('The turbine power matrix should be of dimensions 1 WD X 1 WS X 2 Turbines')
print(turbine_powers)
print("Shape: ",turbine_powers.shape)

# Single wind speed and wind direction
print('\n============================= Single Wind Direction and Multiple Wind Speeds =============================')


wind_speeds = np.array([8.0, 9.0, 10.0])
fi.reinitialize(wind_speeds=wind_speeds)
yaw_angles = np.zeros([1,3,2]) # 1 wind direction, 3 wind speeds, 2 turbines
fi.calculate_wake(yaw_angles=yaw_angles)
turbine_powers = fi.get_turbine_powers()/1000.
print('The turbine power matrix should be of dimensions 1 WD X 3 WS X 2 Turbines')
print(turbine_powers[0,0,:])
print("Shape: ",turbine_powers.shape)


# Single wind speed and wind direction
print('\n============================= Multiple Wind Directions and Multiple Wind Speeds =============================')

wind_directions = np.array([260.])
# wind_directions = np.array([260., 270., 280.])
wind_speeds = np.arange(0.5,30.0,1)
# wind_speeds = np.array([8.0, 9.0, 10.0, 11.0, 13.0])
fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds)
yaw_angles = np.zeros([len(wind_directions),len(wind_speeds),2]) # 1 wind direction, 3 wind speeds, 2 turbines
fi.calculate_wake(yaw_angles=yaw_angles)
turbine_powers = fi.get_turbine_powers()/1000.
print('The turbine power matrix should be of dimensions 3 WD X 3 WS X 2 Turbines')
print(turbine_powers[:,:,:])
print("Shape: ",turbine_powers.shape)

# wind_directions = np.array([260., 270., 280.])
# wind_speeds = np.array([8.0, 9.0, 10.0, 11.0, 13.0])

fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds)
yaw_angles = np.zeros([len(wind_directions),len(wind_speeds),2]) # 1 wind direction, 3 wind speeds, 2 turbines
print('What should Ctp be?',fi.floris.farm.turbine_fCts[0][1](wind_speeds))
CT_same = fi.floris.farm.turbine_fCts[0][1](wind_speeds)
print('CT same',CT_same)
Ct_test = np.zeros((len(wind_directions),len(wind_speeds), 2))
for i in range(len(wind_speeds)):
    Ct_test[:,i,:] = CT_same[i]*0.75
    # Ct_test[:,i,:] = CT_same[i]

print(Ct_test)

atest = 0.5 * (1 - np.sqrt(1 - Ct_test))
cp_test = 0.77*4*atest*(1-atest)**2
print('what would cp be?', fi.floris.farm.turbine_fCps[0][1](wind_speeds))
print('What will cp be?', cp_test)
fi.calculate_wake(yaw_angles=yaw_angles, CT_inputs=Ct_test)
turbine_powers_ct = fi.get_turbine_powers()/1000.
print('The turbine power matrix should be of dimensions 3 WD X 3 WS X 2 Turbines')
print(turbine_powers_ct[:,:,:])
print("Shape: ",turbine_powers_ct.shape)

for i in range(len(wind_speeds)):
    ai_loop = 0.5 * (1-np.sqrt(1 - fi.floris.farm.turbine_fCts[0][1](wind_speeds[i])))
    print('a in between', ai_loop)


# power_to_save = np.zeros((len(wind_speeds), 2))
# power_to_save[:] = turbine_powers_ct[0,:,:]
# np.savetxt('analytic_power_77.txt',power_to_save)
loaded77 = np.loadtxt('C:\\Users\\gstarke\\Documents\\Research_Programs\\FLORIS\\floris\\analytic_power_77.txt')

print(fi.floris.farm.turbine_fCts[0][1])
ct_interped = fi.floris.farm.turbine_fCts[0][1](np.arange(3.5,30,0.25))
a_in_between = 0.5 * (1 - np.sqrt(1 - ct_interped))
cp_fromct = 4*a_in_between*(1 - a_in_between)**2
cp_interped = fi.floris.farm.turbine_fCps[0][1](np.arange(3.5,30,0.25))

plt.figure()
# plt.plot(np.arange(5,30), ct_interped)
plt.plot(np.arange(3.5,30,0.25), cp_interped, '-m', label='Interpolated in floris')
plt.plot(np.arange(3.5,30,0.25), cp_fromct, '-k', label = 'From C_T -> a -> C_P')
plt.plot(np.arange(3.5,30,0.25), cp_fromct*0.8, '--k', label = 'From C_T adjusted using 0.8')
plt.plot(np.arange(3.5,30,0.25), cp_fromct*0.77, '--g', label = 'From C_T adjusted using 0.77')
# plt.plot(np.arange(3.5,30,0.25), cp_interped /cp_fromct, '*b', label='Ratio')
plt.legend()
plt.grid(True)
plt.xlabel('Wind speed [m/s]')
plt.ylabel(r'$C_P$')
plt.show()

plt.figure()
for i in range(len(wind_directions)):
    for j in range(len(wind_speeds)):
        plt.plot(turbine_powers[i,j,:],'k-*')
        plt.plot(turbine_powers_ct[i,j,:],'b-*')
        print(turbine_powers[i,j,:])
        print(turbine_powers_ct[i,j,:])
plt.show()

plt.figure()
# plt.plot(wind_speeds,turbine_powers[0,:,0],'k-*',label='Using floris interpolation')
# plt.plot(wind_speeds,turbine_powers_ct[0,:,0],'r-*', label = 'From analytic C_T -> a -> C_P using 0.8')
# plt.plot(wind_speeds,loaded77[:,0],'b-*', label = 'From analytic C_T -> a -> C_P using 0.77')

# plt.plot(wind_speeds,turbine_powers[0,:,0],'k-',label='Turbine 1')
# plt.plot(wind_speeds,turbine_powers[0,:,1],'k--',label='Turbine 2')
plt.plot(wind_speeds,turbine_powers[0,:,:],'k-*',label='Using floris interpolation')
plt.plot(wind_speeds,turbine_powers_ct[0,:,:],'r-', label = 'From analytic C_T -> a -> C_P using correction')
# plt.plot(wind_speeds,loaded77[:,:],'b-', label = 'From analytic C_T -> a -> C_P using 0.77')

plt.legend()
plt.grid(True)
plt.ylim([-150,5500])
plt.ylabel('Power [kW]')
plt.xlabel('Wind Speed [m/s]')
plt.title('Comparison of the turbines at max power')
plt.show()

plt.figure()
plt.plot(wind_speeds,turbine_powers[0,:,1],'k-*',label='Using floris interpolation')
plt.plot(wind_speeds,turbine_powers_ct[0,:,1],'r-*', label = 'From analytic C_T -> a -> C_P using correction')
# plt.plot(wind_speeds,loaded77[:,1],'b-*', label = 'From analytic C_T -> a -> C_P using 0.77')

plt.legend()
plt.grid(True)
plt.ylim([-150,5500])
plt.ylabel('Power [kW]')
plt.xlabel('Wind Speed [m/s]')
plt.title('Comparison of the Second turbines at max power')
plt.show()

# plt.figure()
# # plt.plot(np.arange(5,30), ct_interped)
# plt.plot(np.arange(5,30,0.25),cp_interped /cp_fromct, label='ratio')
# plt.legend()
# plt.show()

# print('averave offset', np.mean(cp_interped[cp_interped>0] /cp_fromct[cp_interped>0]))