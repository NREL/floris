# FLORIS driver program

# import specific models
from turbines.NREL5MW import NREL5MW
from wakes.JensenJimenez import JensenJimenez
from src.models.WakeCombination import WakeCombination
from farms.TwoByTwo import TwoByTwo
from src.io.InputReader import InputReader


inputReader = InputReader()

# turbine input
turbineInput = "turbines/NREL5MW.json"
turbine = inputReader.buildTurbine(turbineInput)

# wake input
wakeInput = "wakes/JensenJimenez.json"

# farm input
farmInput = "farms/TwoByTwo.json"
# TODO: add controls to farm

twobytwo = TwoByTwo(turbine=turbine,
                    wake=JensenJimenez(),
                    combination=WakeCombination("fls"))

t0 = twobytwo.getTurbineAtCoord((0,0))

print("t0.Cp", t0.Cp)
print("t0.Ct", t0.Ct)
print("t0.power", t0.power)
print("t0.aI", t0.aI)
print("t0.get_average_velocity()", t0.get_average_velocity())

ff = twobytwo.get_flow_field()

ff.plot_flow_field_plane()
