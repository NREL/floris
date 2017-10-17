import os
import sys

# import specific models
from turbines.NREL5MW import NREL5MW
from wakes.Jimenez_Floris_FLS import Jimenez_Floris_FLS

# construct the turbines
# wake: jimenez deflection - floris velocity - freestream linear superposition
jimenez_floris_fls = Jimenez_Floris_FLS()

# turbine: NREL 5MW
nrelfiveMW = NREL5MW(jimenez_floris_fls)

wake = nrelfiveMW.getWake()
velocity = wake.getVelocity()

# print(velocity.getType())
