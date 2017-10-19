# FLORIS driver program

# import specific models
from turbines.NREL5MW import NREL5MW
from wakes.Jimenez_Floris_FLS import Jimenez_Floris_FLS
from src.models.WakeCombination import WakeCombination
from farms.TwoByTwo import TwoByTwo
from src.models.FlowField import FlowField

# construct the objects for this simulation

# turbine: NREL 5MW
nrelfiveMW = NREL5MW()

# flow field: FLS combination
ff = FlowField(wakeCombination=WakeCombination("fls"))

# farm: 2 by 2 grid with constant turbine; FLS combination
twobytwo = TwoByTwo(turbine=nrelfiveMW,
                    wake=Jimenez_Floris_FLS())
