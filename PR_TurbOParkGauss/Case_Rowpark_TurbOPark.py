import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from floris import FlorisModel

fmodel1 = FlorisModel("Case_RowPark_TurbOPark.yaml")
fmodel1.run()
raws1 = fmodel1.turbine_average_velocities

u0 = fmodel1.wind_speeds

fmodel2 = FlorisModel("Case_RowPark_TurbOParkGauss.yaml")
fmodel2.run()
raws2 = fmodel2.turbine_average_velocities

df = pd.read_csv("Rowpark_Orsted.csv")

fig, ax = plt.subplots()
ax.scatter(range(1,11),raws1/u0,s=80,marker='p',label='Floris - TurbOPark')
ax.scatter(range(1,11),raws2/u0,s=80,marker='^',label='Floris - TurbOPark_Gauss')
ax.plot(df.values[:,0],df.values[:,1],'or',label='Orsted - TurbOPark')
ax.set_xlabel('Turbine number [-]')
ax.set_ylabel('Normalized waked wind speed')
ax.set_xlim(0,11)
ax.set_ylim(0.25,1.05)
ax.legend()

plt.show()
