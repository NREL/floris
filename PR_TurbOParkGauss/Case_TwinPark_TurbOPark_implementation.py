import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from floris import (
    FlorisModel,
    TimeSeries,
)


fmodel = FlorisModel("Case_TwinPark_TurbOPark.yaml")

wd_array = np.arange(225,315,0.1)
fmodel.set(
    wind_data=TimeSeries(
        wind_speeds=8.0, wind_directions=wd_array, turbulence_intensities=0.06
    )
)

fmodel.run()

rawss = fmodel.turbine_average_velocities
raws = rawss[:,1]
u0 = fmodel.wind_speeds

###

fmodel2 = FlorisModel("Case_TwinPark_TurbOParkGauss.yaml")

wd_array = np.arange(225,315,0.1)
fmodel2.set(
    wind_data=TimeSeries(
        wind_speeds=8.0, wind_directions=wd_array, turbulence_intensities=0.06
    )
)

fmodel2.run()

rawss2 = fmodel2.turbine_average_velocities
raws2 = rawss2[:,1]

###
df = pd.read_csv("/mnt/c/Users/Jasper.Kreeft/Data/PYTHON/NREL/floris4/PR_TurbOParkGauss/WindDirection_Sweep_Orsted.csv")

###
fig, ax1 = plt.subplots()
ax1.plot(wd_array,raws/u0,label='Floris 4 - TurbOPark')
ax1.plot(wd_array,raws2/u0,label='Floris 4 - TurbOPark-Gauss')
ax1.plot(df.values[:,0],df.values[:,1],'--',label='Orsted - TurbOPark')

ax1.set_xlabel('Wind Direction (deg)')
ax1.set_ylabel('Normalized rotor averaged waked wind speed [-]')
ax1.set_xlim(240,300)
ax1.set_ylim(0.65,1.05)
ax1.legend()
plt.show()
