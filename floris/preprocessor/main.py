
from floris.preprocessor.v1_0_0 import V1_0_0
from floris.preprocessor.v2_0_0 import V2_0_0

if __name__=="__main__":
    v100 = V1_0_0()
    v100.export(filename="v1.0.0.json")
    v200 = V2_0_0(
        v100.meta_dict,
        v100.turbine_dict,
        v100.wake_dict,
        v100.farm_dict
    )
    v200.export(filename="v2.0.0.json")
