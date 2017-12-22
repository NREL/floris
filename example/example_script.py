import sys
sys.path.append('../floris')
from floris import floris

floris = floris()
floris.process_input("floris.json")

for coord, turbine in floris.farm.turbine_map.items():
    print(str(coord) + ":")
    print("\tCp -", turbine.Cp)
    print("\tCt -", turbine.Ct)
    print("\tpower -", turbine.power)
    print("\tai -", turbine.aI)
    print("\taverage velocity -", turbine.get_average_velocity())

floris.farm.flow_field.plot_z_planes([0.2, 0.5, 0.8])
