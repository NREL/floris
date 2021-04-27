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


import pytest


def turbines_to_array(turbine_list: list):
    return [[t.Ct, t.power, t.aI, t.average_velocity] for t in turbine_list]


def print_test_values(turbine_list: list):
    for t in turbine_list:
        print(
            "({:.7f}, {:.7f}, {:.7f}, {:.7f}),".format(
                t.Ct, t.power, t.aI, t.average_velocity
            )
        )


@pytest.fixture
def sample_inputs_fixture():
    return SampleInputs()


class SampleInputs:
    """
    SampleInputs class
    """

    def __init__(self):
        self.turbine = {
            "type": "turbine",
            "name": "nrel_5mw",
            "description": "NREL 5MW",
            "properties": {
                "rotor_diameter": 126.0,
                "hub_height": 90.0,
                "blade_count": 3,
                "pP": 1.88,
                "pT": 1.88,
                "generator_efficiency": 1.0,
                "power_thrust_table": {
                    "power": [
                        0.0,
                        0.15643578,
                        0.31287155,
                        0.41306749,
                        0.44895632,
                        0.46155227,
                        0.46330747,
                        0.46316077,
                        0.46316077,
                        0.46280642,
                        0.45223111,
                        0.39353012,
                        0.3424487,
                        0.2979978,
                        0.25931677,
                        0.22565665,
                        0.19636572,
                        0.17087684,
                        0.1486965,
                        0.12939524,
                        0.11259934,
                        0.0979836,
                        0.08526502,
                        0.07419736,
                        0.06456631,
                        0.05618541,
                        0.04889237,
                        0.0,
                    ],
                    "thrust": [
                        1.10610965,
                        1.09515807,
                        1.0227122,
                        0.9196487,
                        0.85190470,
                        0.80328229,
                        0.76675469,
                        0.76209299,
                        0.76209299,
                        0.75083241,
                        0.67210674,
                        0.52188504,
                        0.43178758,
                        0.36443258,
                        0.31049874,
                        0.26696686,
                        0.22986909,
                        0.19961578,
                        0.17286245,
                        0.15081457,
                        0.13146666,
                        0.11475968,
                        0.10129584,
                        0.0880188,
                        0.07746819,
                        0.06878621,
                        0.05977061,
                        0.0,
                    ],
                    "wind_speed": [
                        0.0,
                        2.5,
                        3.52338654,
                        4.57015961,
                        5.61693268,
                        6.66370575,
                        7.71047882,
                        8.75725189,
                        9.80402496,
                        10.85079803,
                        11.70448774,
                        12.25970155,
                        12.84125247,
                        13.45038983,
                        14.08842222,
                        14.75672029,
                        15.45671974,
                        16.18992434,
                        16.95790922,
                        17.76232421,
                        18.60489742,
                        19.48743891,
                        20.41184461,
                        21.38010041,
                        22.39428636,
                        23.45658122,
                        24.56926707,
                        30.0,
                    ],
                },
                "yaw_angle": 0.0,
                "tilt_angle": 0.0,
                "TSR": 8.0,
            },
        }

        self.farm = {
            "type": "farm",
            "name": "farm_example_2x2",
            "properties": {
                "wind_speed": [8.0],
                "wind_direction": [270.0],
                "turbulence_intensity": [0.1],
                "wind_shear": 0.12,
                "wind_veer": 0.0,
                "air_density": 1.225,
                "wake_combination": "sosfs",
                "layout_x": [
                    0.0,
                    5 * self.turbine["properties"]["rotor_diameter"],
                    10 * self.turbine["properties"]["rotor_diameter"],
                ],
                "layout_y": [0.0, 0.0, 0.0],
                "wind_x": [0],
                "wind_y": [0],
                "specified_wind_height": self.turbine["properties"]["hub_height"],
            },
        }

        self.wake = {
            "type": "wake",
            "name": "wake_default",
            "properties": {
                "velocity_model": "gauss_legacy",
                "deflection_model": "gauss",
                "combination_model": "sosfs",
                "turbulence_model": "crespo_hernandez",
                "parameters": {
                    "wake_deflection_parameters": {
                        "gauss": {
                            "dm": 1.0,
                            "eps_gain": 0.2,
                            "use_secondary_steering": False,
                        }
                    },
                    "wake_velocity_parameters": {
                        "gauss_legacy": {
                            "calculate_VW_velocities": False,
                            "eps_gain": 0.2,
                            "ka": 0.38,
                            "kb": 0.004,
                            "use_yaw_added_recovery": False,
                        }
                    },
                },
            },
        }

        self.floris = {
            "farm": self.farm,
            "turbine": self.turbine,
            "wake": self.wake,
            "logging": {
                "console": {"enable": True, "level": 1},
                "file": {"enable": False, "level": 1},
            },
        }
