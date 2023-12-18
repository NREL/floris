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

import os
from pathlib import Path

import numpy as np
import yaml

from floris.tools import build_turbine_dict, check_smooth_power_curve

def test_build_turbine_dict():

    orig_file_path = Path(__file__).resolve().parent / "data" / "nrel_5MW_v3legacy.yaml"
    test_turb_name = "test_turbine_export"
    test_file_path = "."

    in_dict = yaml.safe_load( open(orig_file_path, "r") )

    # Mocked up turbine data
    turbine_data_dict = {
        "wind_speed":in_dict["power_thrust_table"]["wind_speed"],
        "power_coefficient":in_dict["power_thrust_table"]["power"],
        "thrust_coefficient":in_dict["power_thrust_table"]["thrust"]
    }

    build_turbine_dict(
        turbine_data_dict,
        test_turb_name,
        file_name=os.path.join(test_file_path, test_turb_name+".yaml"),
        generator_efficiency=in_dict["generator_efficiency"],
        hub_height=in_dict["hub_height"],
        pP=in_dict["pP"],
        pT=in_dict["pT"],
        rotor_diameter=in_dict["rotor_diameter"],
        TSR=in_dict["TSR"],
        ref_air_density=in_dict["ref_density_cp_ct"],
        ref_tilt=in_dict["ref_tilt_cp_ct"]
    )

    test_dict = yaml.safe_load(
        open(os.path.join(test_file_path, test_turb_name+".yaml"), "r")
    )

    # Get absolute values
    Cp = np.array(in_dict["power_thrust_table"]["power"])
    Ct = np.array(in_dict["power_thrust_table"]["thrust"])
    ws = np.array(in_dict["power_thrust_table"]["wind_speed"])

    P = 0.5 * in_dict["ref_density_cp_ct"] * (np.pi * in_dict["rotor_diameter"]**2/4) \
        * Cp * ws**3
    T = 0.5 * in_dict["ref_density_cp_ct"] * (np.pi * in_dict["rotor_diameter"]**2/4) \
        * Ct * ws**2

    # Correct intended difference for test; assert equal or close
    in_dict["power_thrust_table"]["power"] = list(P / 1000)
    in_dict["power_thrust_table"]["thrust_coefficient"] = in_dict["power_thrust_table"]["thrust"]
    in_dict["power_thrust_table"].pop("thrust")
    in_dict["ref_air_density"] = in_dict["ref_density_cp_ct"]
    in_dict.pop("ref_density_cp_ct")
    in_dict["ref_tilt"] = in_dict["ref_tilt_cp_ct"]
    in_dict.pop("ref_tilt_cp_ct")
    test_dict["turbine_type"] = in_dict["turbine_type"]
    assert set(in_dict.keys()) == set(test_dict.keys())
    assert np.allclose(
        in_dict["power_thrust_table"]["power"],
        in_dict["power_thrust_table"]["power"]
    )

    turbine_data_dict = {
        "wind_speed":in_dict["power_thrust_table"]["wind_speed"],
        "power": P/1000,
        "thrust": T/1000
    }

    build_turbine_dict(
        turbine_data_dict,
        test_turb_name,
        file_name=os.path.join(test_file_path, test_turb_name+".yaml"),
        generator_efficiency=in_dict["generator_efficiency"],
        hub_height=in_dict["hub_height"],
        pP=in_dict["pP"],
        pT=in_dict["pT"],
        rotor_diameter=in_dict["rotor_diameter"],
        TSR=in_dict["TSR"],
        ref_air_density=in_dict["ref_air_density"],
        ref_tilt=in_dict["ref_tilt"]
    )

    test_dict = yaml.safe_load(
        open(os.path.join(test_file_path, test_turb_name+".yaml"), "r")
    )

    test_dict["turbine_type"] = in_dict["turbine_type"]
    assert set(in_dict.keys()) == set(test_dict.keys())
    for k in in_dict.keys():
        if type(in_dict[k]) is dict:
            for k2 in in_dict[k].keys():
                assert np.allclose(in_dict[k][k2], test_dict[k][k2])
        elif type(in_dict[k]) is str:
            assert in_dict[k] == test_dict[k]
        else:
            assert np.allclose(in_dict[k], test_dict[k])

    os.remove( os.path.join(test_file_path, test_turb_name+".yaml") )

def test_check_smooth_power_curve():

    p1 = np.array([0, 1, 2, 3, 3, 3, 3, 2, 1], dtype=float)*1000 # smooth
    p2 = np.array([0, 1, 2, 3, 2.99, 3.01, 3, 2, 1], dtype=float)*1000 # non-smooth
    
    p3 = p1.copy()
    p3[5] = p3[5] + 9e-4  # just smooth enough

    p4 = p1.copy()
    p4[5] = p4[5] + 1.1e-3 # just not smooth enough

    # Without a shutdown region
    p5 = p1[:-3] # smooth
    p6 = p2[:-3] # non-smooth

    assert check_smooth_power_curve(p1)
    assert not check_smooth_power_curve(p2)
    assert check_smooth_power_curve(p3)
    assert not check_smooth_power_curve(p4)
    assert check_smooth_power_curve(p5)
    assert not check_smooth_power_curve(p6)
