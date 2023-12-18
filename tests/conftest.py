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

from __future__ import annotations

import copy

import numpy as np
import pytest

from floris.simulation import (
    Floris,
    FlowField,
    FlowFieldGrid,
    PointsGrid,
    TurbineGrid,
)


def turbines_to_array(turbine_list: list):
    return [[t.average_velocity, t.Ct, t.power, t.axial_induction] for t in turbine_list]


def assert_results_arrays(test: np.array, baseline: np.array):
    if np.shape(test) != np.shape(baseline):
        raise ValueError("test and baseline results have mismatched shapes.")

    for test_dim0, baseline_dim0 in zip(test, baseline):
        for test_dim1, baseline_dim1 in zip(test_dim0, baseline_dim0):
            assert np.allclose(test_dim1, baseline_dim1)


def assert_results(test: list, baseline: list):
    if len(test) != len(baseline):
        raise ValueError("assert_results: test and baseline results have mismatched lengths.")

    for i in range(len(test)):
        for j, (t, b) in enumerate(zip(test[i], baseline[i])):
            # print(j, t, b)
            assert t == pytest.approx(b)


def print_test_values(
    average_velocities: list,
    thrusts: list,
    powers: list,
    axial_inductions: list,
    max_findex_print: int | None =None
):
    n_findex, n_turb = np.shape(average_velocities)
    if max_findex_print is not None:
        n_findex = min(n_findex, max_findex_print)
    for i in range(n_findex):
        print("[")
        for j in range(n_turb):
            print(
                "    [{:.7f}, {:.7f}, {:.7f}, {:.7f}],".format(
                    average_velocities[i,j], thrusts[i,j], powers[i,j],
                    axial_inductions[i,j]
                )
            )
        print("],")


WIND_DIRECTIONS = [
    270.0,
    270.0,
    270.0,
    270.0,
    360.0,
    360.0,
    360.0,
    360.0,
    285.0,
    285.0,
    285.0,
    285.0,
    315.0,
    315.0,
    315.0,
    315.0,
]
WIND_SPEEDS = [
    8.0,
    9.0,
    10.0,
    11.0,
    8.0,
    9.0,
    10.0,
    11.0,
    8.0,
    9.0,
    10.0,
    11.0,
    8.0,
    9.0,
    10.0,
    11.0,
]

# FINDEX is the length of the number of conditions, so it can be
# len(WIND_DIRECTIONS) or len(WIND_SPEEDS
N_FINDEX = len(WIND_DIRECTIONS)

X_COORDS = [
    0.0,
    5 * 126.0,
    10 * 126.0
]
Y_COORDS = [
    0.0,
    0.0,
    0.0
]
Z_COORDS = [
    90.0,
    90.0,
    90.0
]
N_TURBINES = len(X_COORDS)
ROTOR_DIAMETER = 126.0
TURBINE_GRID_RESOLUTION = 2
TIME_SERIES = False


## Unit test fixtures

@pytest.fixture
def flow_field_fixture(sample_inputs_fixture):
    flow_field_dict = sample_inputs_fixture.flow_field
    return FlowField.from_dict(flow_field_dict)

@pytest.fixture
def turbine_grid_fixture(sample_inputs_fixture) -> TurbineGrid:
    turbine_coordinates = np.array(list(zip(X_COORDS, Y_COORDS, Z_COORDS)))
    rotor_diameters = ROTOR_DIAMETER * np.ones( (N_TURBINES) )
    return TurbineGrid(
        turbine_coordinates=turbine_coordinates,
        turbine_diameters=rotor_diameters,
        wind_directions=np.array(WIND_DIRECTIONS),
        grid_resolution=TURBINE_GRID_RESOLUTION,
        time_series=TIME_SERIES
    )

@pytest.fixture
def flow_field_grid_fixture(sample_inputs_fixture) -> FlowFieldGrid:
    turbine_coordinates = np.array(list(zip(X_COORDS, Y_COORDS, Z_COORDS)))
    rotor_diameters = ROTOR_DIAMETER * np.ones( (N_FINDEX, N_TURBINES) )
    return FlowFieldGrid(
        turbine_coordinates=turbine_coordinates,
        turbine_diameters=rotor_diameters,
        wind_directions=np.array(WIND_DIRECTIONS),
        grid_resolution=[3,2,2]
    )

@pytest.fixture
def points_grid_fixture(sample_inputs_fixture) -> PointsGrid:
    turbine_coordinates = np.array(list(zip(X_COORDS, Y_COORDS, Z_COORDS)))
    rotor_diameters = ROTOR_DIAMETER * np.ones( (N_FINDEX, N_TURBINES) )
    points_x = [0.0, 10.0]
    points_y = [0.0, 0.0]
    points_z = [1.0, 2.0]
    return PointsGrid(
        turbine_coordinates=turbine_coordinates,
        turbine_diameters=rotor_diameters,
        wind_directions=np.array(WIND_DIRECTIONS),
        grid_resolution=None,
        time_series=False,
        points_x=points_x,
        points_y=points_y,
        points_z=points_z,
    )

@pytest.fixture
def floris_fixture():
    sample_inputs = SampleInputs()
    return Floris(sample_inputs.floris)

@pytest.fixture
def sample_inputs_fixture():
    return SampleInputs()


class SampleInputs:
    """
    SampleInputs class
    """

    def __init__(self):
        self.turbine = {
            "turbine_type": "nrel_5mw",
            "rotor_diameter": 126.0,
            "hub_height": 90.0,
            "pP": 1.88,
            "pT": 1.88,
            "generator_efficiency": 1.0,
            "ref_air_density": 1.225,
            "ref_tilt": 5.0,
            "power_thrust_table": {
                "power": [
                    0.0,
                    0.0,
                    40.5,
                    177.7,
                    403.9,
                    737.6,
                    1187.2,
                    1771.1,
                    2518.6,
                    3448.41,
                    3552.15,
                    3657.95,
                    3765.16,
                    3873.95,
                    3984.49,
                    4096.56,
                    4210.69,
                    4326.15,
                    4443.41,
                    4562.51,
                    4683.43,
                    4806.18,
                    4929.92,
                    5000.37,
                    5000.02,
                    5000.0,
                    4999.99,
                    5000.0,
                    5000.0,
                    5000.0,
                    5000.0,
                    5000.0,
                    5000.0,
                    5000.0,
                    5000.0,
                    5000.0,
                    5000.0,
                    5000.0,
                    5000.0,
                    5000.0,
                    5000.0,
                    5000.0,
                    5000.0,
                    0.0,
                    0.0
                ],
                "thrust_coefficient": [
                    0.0,
                    0.0,
                    2.497990147,
                    1.766833378,
                    1.408360153,
                    1.201348494,
                    1.065133759,
                    0.977936955,
                    0.936281559,
                    0.905425262,
                    0.902755344,
                    0.90016155,
                    0.895745235,
                    0.889630636,
                    0.883651878,
                    0.877788261,
                    0.872068513,
                    0.866439424,
                    0.860930874,
                    0.855544522,
                    0.850276473,
                    0.845148048,
                    0.840105118,
                    0.811165614,
                    0.764009698,
                    0.728584172,
                    0.698944675,
                    0.672754103,
                    0.649082557,
                    0.627368152,
                    0.471373796,
                    0.372703289,
                    0.30290131,
                    0.251235686,
                    0.211900735,
                    0.181210571,
                    0.156798163,
                    0.137091212,
                    0.120753164,
                    0.106941036,
                    0.095319286,
                    0.085631997,
                    0.077368152,
                    0.0,
                    0.0
                ],
                "wind_speed": [
                    0.0,
                    2.9,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    10.1,
                    10.2,
                    10.3,
                    10.4,
                    10.5,
                    10.6,
                    10.7,
                    10.8,
                    10.9,
                    11.0,
                    11.1,
                    11.2,
                    11.3,
                    11.4,
                    11.5,
                    11.6,
                    11.7,
                    11.8,
                    11.9,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                    16.0,
                    17.0,
                    18.0,
                    19.0,
                    20.0,
                    21.0,
                    22.0,
                    23.0,
                    24.0,
                    25.0,
                    25.01,
                    50.0
                ],
            },
            "TSR": 8.0
        }

        self.turbine_floating = copy.deepcopy(self.turbine)
        self.turbine_floating["floating_tilt_table"] = {
            "tilt": [
                5.0,
                5.0,
                5.0,
            ],
            "wind_speed": [
                0.0,
                25.0,
                50.0,
            ],
        }
        self.turbine_floating["correct_cp_ct_for_tilt"] = True

        self.turbine_multi_dim = copy.deepcopy(self.turbine)
        del self.turbine_multi_dim['power_thrust_table']
        self.turbine_multi_dim["multi_dimensional_cp_ct"] = True
        self.turbine_multi_dim["power_thrust_data_file"] = ""

        self.farm = {
            "layout_x": X_COORDS,
            "layout_y": Y_COORDS,
            "turbine_type": [self.turbine]
        }

        self.flow_field = {
            "wind_speeds": WIND_SPEEDS,
            "wind_directions": WIND_DIRECTIONS,
            "turbulence_intensity": 0.1,
            "wind_shear": 0.12,
            "wind_veer": 0.0,
            "air_density": 1.225,
            "reference_wind_height": self.turbine["hub_height"],
        }

        self.wake = {
            "model_strings": {
                "velocity_model": "jensen",
                "deflection_model": "jimenez",
                "combination_model": "sosfs",
                "turbulence_model": "crespo_hernandez",
            },
            "wake_deflection_parameters": {
                "gauss": {
                    "ad": 0.0,
                    "alpha": 0.58,
                    "bd": 0.0,
                    "beta": 0.077,
                    "dm": 1.0,
                    "ka": 0.38,
                    "kb": 0.004
                },
                "jimenez": {
                    "ad": 0.0,
                    "bd": 0.0,
                    "kd": 0.05,
                },
                "empirical_gauss": {
                   "horizontal_deflection_gain_D": 3.0,
                   "vertical_deflection_gain_D": -1,
                   "deflection_rate": 30,
                   "mixing_gain_deflection": 0.0,
                   "yaw_added_mixing_gain": 0.0
                },
            },
            "wake_velocity_parameters": {
                "gauss": {
                    "alpha": 0.58,
                    "beta": 0.077,
                    "ka": 0.38,
                    "kb": 0.004
                },
                "jensen": {
                    "we": 0.05,
                },
                "cc": {
                    "a_s": 0.179367259,
                    "b_s": 0.0118889215,
                    "c_s1": 0.0563691592,
                    "c_s2": 0.13290157,
                    "a_f": 3.11,
                    "b_f": -0.68,
                    "c_f": 2.41,
                    "alpha_mod": 1.0
                },
                "turbopark": {
                    "A": 0.04,
                    "sigma_max_rel": 4.0
                },
                "empirical_gauss": {
                    "wake_expansion_rates": [0.023, 0.008],
                    "breakpoints_D": [10],
                    "sigma_0_D": 0.28,
                    "smoothing_length_D": 2.0,
                    "mixing_gain_velocity": 2.0
                },
            },
            "wake_turbulence_parameters": {
                "crespo_hernandez": {
                    "initial": 0.1,
                    "constant": 0.5,
                    "ai": 0.8,
                    "downstream": -0.32
                },
                "wake_induced_mixing": {
                    "atmospheric_ti_gain": 0.0
                }
            },
            "enable_secondary_steering": False,
            "enable_yaw_added_recovery": False,
            "enable_transverse_velocities": False,
        }

        self.floris = {
            "farm": self.farm,
            "flow_field": self.flow_field,
            "wake": self.wake,
            "solver": {
                "type": "turbine_grid",
                "turbine_grid_points": 3,
            },
            "logging": {
                "console": {"enable": True, "level": 1},
                "file": {"enable": False, "level": 1},
            },
            "name": "conftest",
            "description": "Inputs used for testing",
            "floris_version": "v3.0.0",
        }
