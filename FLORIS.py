# FLORIS driver program

# import specific models
from turbines.NREL5MW import NREL5MW
from wakes.JensenJimenez import JensenJimenez
from src.models.WakeCombination import WakeCombination
from farms.TwoByTwo import TwoByTwo


# farm: 2 by 2 staggered grid with NREL 5WM turbine; FLS combination
twobytwo = TwoByTwo(turbine=NREL5MW(),
                    wake=JensenJimenez(),
                    combination=WakeCombination("fls"))

t0 = twobytwo.getTurbineAtCoord((0,0))

print("t0.Cp", t0.Cp)
print("t0.Ct", t0.Ct)
print("t0.power", t0.power)
print("t0.aI", t0.aI)
print("t0.get_average_velocity()", t0.get_average_velocity())
