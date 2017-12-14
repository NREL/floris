import sys
sys.path.append('../floris')
from floris import floris

floris = floris()
floris.process_input("floris.json")

for coord in floris.farm.get_turbine_coords():
    turbine = floris.farm.get_turbine_at_coord(coord)
    print(str(coord) + ":")
    print("\tCp -", turbine.Cp)
    print("\tCt -", turbine.Ct)
    print("\tpower -", turbine.power)
    print("\tai -", turbine.aI)
    print("\taverage velocity -", turbine.get_average_velocity())
    #print(turbine.grid)
    #print(turbine.velocities)

floris.farm.flow_field.plot_flow_field_planes()
