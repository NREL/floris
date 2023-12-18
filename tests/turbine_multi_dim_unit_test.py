# Copyright 2023 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from floris.simulation import (
    Turbine,
    TurbineMultiDimensional,
)
from floris.simulation.turbine_multi_dim import (
    axial_induction_multidim,
    Ct_multidim,
    multidim_Ct_down_select,
    multidim_power_down_select,
    MultiDimensionalPowerThrustTable,
    power_multidim,
)
from tests.conftest import SampleInputs, WIND_SPEEDS


TEST_DATA = Path(__file__).resolve().parent.parent / "floris" / "turbine_library"
CSV_INPUT = TEST_DATA / "iea_15MW_multi_dim_Tp_Hs.csv"


# size 16 x 1 x 1 x 1
# 16 wind speed and wind direction combinations from conftest
WIND_CONDITION_BROADCAST = np.reshape(np.array(WIND_SPEEDS), (-1, 1, 1, 1))

INDEX_FILTER = [0, 2]


def test_multidim_Ct_down_select():
    CONDITIONS = {'Tp': 2, 'Hs': 1}

    turbine_data = SampleInputs().turbine_multi_dim
    turbine_data["power_thrust_data_file"] = CSV_INPUT
    turbine = TurbineMultiDimensional.from_dict(turbine_data)

    downselect_turbine_fCts = multidim_Ct_down_select([[turbine.fCt_interp]], CONDITIONS)

    assert downselect_turbine_fCts == turbine.fCt_interp[(2, 1)]


def test_multidim_power_down_select():
    CONDITIONS = {'Tp': 2, 'Hs': 1}

    turbine_data = SampleInputs().turbine_multi_dim
    turbine_data["power_thrust_data_file"] = CSV_INPUT
    turbine = TurbineMultiDimensional.from_dict(turbine_data)

    downselect_power_interps = multidim_power_down_select([[turbine.power_interp]], CONDITIONS)

    assert downselect_power_interps == turbine.power_interp[(2, 1)]


def test_multi_dimensional_power_thrust_table():
    turbine_data = SampleInputs().turbine_multi_dim
    turbine_data["power_thrust_data_file"] = CSV_INPUT
    df_data = pd.read_csv(turbine_data["power_thrust_data_file"])
    flattened_dict = MultiDimensionalPowerThrustTable.from_dataframe(df_data)
    flattened_dict_base = {
        ('Tp', '2', 'Hs', '1'): [],
        ('Tp', '2', 'Hs', '5'): [],
        ('Tp', '4', 'Hs', '1'): [],
        ('Tp', '4', 'Hs', '5'): [],
    }
    assert flattened_dict == flattened_dict_base

    # Test for initialization errors
    for el in ("ws", "Cp", "Ct"):
        df_data = pd.read_csv(turbine_data["power_thrust_data_file"])
        df = df_data.drop(el, axis=1)
        with pytest.raises(ValueError):
            MultiDimensionalPowerThrustTable.from_dataframe(df)


def test_turbine_init():
    turbine_data = SampleInputs().turbine_multi_dim
    turbine_data["power_thrust_data_file"] = CSV_INPUT
    turbine = TurbineMultiDimensional.from_dict(turbine_data)
    assert turbine.rotor_diameter == turbine_data["rotor_diameter"]
    assert turbine.hub_height == turbine_data["hub_height"]
    assert turbine.pP == turbine_data["pP"]
    assert turbine.pT == turbine_data["pT"]
    assert turbine.generator_efficiency == turbine_data["generator_efficiency"]

    assert isinstance(turbine.power_thrust_data, dict)
    assert isinstance(turbine.fCt_interp, dict)
    assert isinstance(turbine.power_interp, dict)
    assert turbine.rotor_radius == turbine_data["rotor_diameter"] / 2.0


def test_ct():
    N_TURBINES = 4

    turbine_data = SampleInputs().turbine_multi_dim
    turbine_data["power_thrust_data_file"] = CSV_INPUT
    turbine = TurbineMultiDimensional.from_dict(turbine_data)
    turbine_type_map = np.array(N_TURBINES * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, :]

    # Single turbine
    # yaw angle / fCt are (n wind direction, n wind speed, n turbine)
    wind_speed = 10.0
    thrust = Ct_multidim(
        velocities=wind_speed * np.ones((1, 1, 3, 3)),
        yaw_angle=np.zeros((1, 1)),
        tilt_angle=np.ones((1, 1)) * 5.0,
        ref_tilt_cp_ct=np.ones((1, 1)) * 5.0,
        fCt=np.array([[turbine.fCt_interp[(2, 1)]]]),
        tilt_interp={turbine.turbine_type: None},
        correct_cp_ct_for_tilt=np.array([[False]]),
        turbine_type_map=turbine_type_map[:,0]
    )

    np.testing.assert_allclose(thrust, np.array([[0.77853469]]))

    # Multiple turbines with index filter
    # 4 turbines with 3 x 3 grid arrays
    thrusts = Ct_multidim(
        velocities=np.ones((N_TURBINES, 3, 3)) * WIND_CONDITION_BROADCAST,  # 16 x 4 x 3 x 3
        yaw_angle=np.zeros((1, N_TURBINES)),
        tilt_angle=np.ones((1, N_TURBINES)) * 5.0,
        ref_tilt_cp_ct=np.ones((1, N_TURBINES)) * 5.0,
        fCt=np.tile(
            [turbine.fCt_interp[(2, 1)]],
            (
                np.shape(WIND_CONDITION_BROADCAST)[0],
                N_TURBINES,
            )
        ),
        tilt_interp={turbine.turbine_type: None},
        correct_cp_ct_for_tilt=np.array([[False] * N_TURBINES]),
        turbine_type_map=turbine_type_map,
        ix_filter=INDEX_FILTER,
    )
    assert len(thrusts[0]) == len(INDEX_FILTER)

    thrusts_truth = np.array([
        [0.77853469, 0.77853469],
        [0.77853469, 0.77853469],
        [0.77853469, 0.77853469],
        [0.6957943,  0.6957943 ],

        [0.77853469, 0.77853469],
        [0.77853469, 0.77853469],
        [0.77853469, 0.77853469],
        [0.6957943,  0.6957943 ],

        [0.77853469, 0.77853469],
        [0.77853469, 0.77853469],
        [0.77853469, 0.77853469],
        [0.6957943,  0.6957943 ],

        [0.77853469, 0.77853469],
        [0.77853469, 0.77853469],
        [0.77853469, 0.77853469],
        [0.6957943,  0.6957943 ],
    ])
    np.testing.assert_allclose(thrusts, thrusts_truth)


def test_power():
    N_TURBINES = 4
    AIR_DENSITY = 1.225

    turbine_data = SampleInputs().turbine_multi_dim
    turbine_data["power_thrust_data_file"] = CSV_INPUT
    turbine = TurbineMultiDimensional.from_dict(turbine_data)
    turbine_type_map = np.array(N_TURBINES * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, :]

    # Single turbine
    wind_speed = 10.0
    p = power_multidim(
        ref_density_cp_ct=AIR_DENSITY,
        rotor_effective_velocities=wind_speed * np.ones((1, 1, 3, 3)),
        power_interp=np.array([[turbine.power_interp[(2, 1)]]]),
    )

    power_truth = [
        [
            [
                [3215682.686486, 3215682.686486, 3215682.686486],
                [3215682.686486, 3215682.686486, 3215682.686486],
                [3215682.686486, 3215682.686486, 3215682.686486],
            ]
        ]
    ]

    np.testing.assert_allclose(p, power_truth )

    # Multiple turbines with ix filter
    rotor_effective_velocities = np.ones((N_TURBINES, 3, 3)) * WIND_CONDITION_BROADCAST
    p = power_multidim(
        ref_density_cp_ct=AIR_DENSITY,
        rotor_effective_velocities=rotor_effective_velocities,
        power_interp=np.tile(
            [turbine.power_interp[(2, 1)]],
            (
                np.shape(WIND_CONDITION_BROADCAST)[0],
                N_TURBINES,
            )
        ),
        ix_filter=INDEX_FILTER,
    )
    assert len(p[0]) == len(INDEX_FILTER)

    power_truth = turbine.power_interp[(2, 1)](rotor_effective_velocities) * AIR_DENSITY
    np.testing.assert_allclose(p, power_truth[:, INDEX_FILTER[0]:INDEX_FILTER[1]])


def test_axial_induction():

    N_TURBINES = 4

    turbine_data = SampleInputs().turbine_multi_dim
    turbine_data["power_thrust_data_file"] = CSV_INPUT
    turbine = TurbineMultiDimensional.from_dict(turbine_data)
    turbine_type_map = np.array(N_TURBINES * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, :]

    baseline_ai = 0.2646995

    # Single turbine
    wind_speed = 10.0
    ai = axial_induction_multidim(
        velocities=wind_speed * np.ones((1, 1, 3, 3)),
        yaw_angle=np.zeros((1, 1)),
        tilt_angle=np.ones((1, 1)) * 5.0,
        ref_tilt_cp_ct=np.ones((1, 1)) * 5.0,
        fCt=np.array([[turbine.fCt_interp[(2, 1)]]]),
        tilt_interp={turbine.turbine_type: None},
        correct_cp_ct_for_tilt=np.array([[False]]),
        turbine_type_map=turbine_type_map[0,0],
    )
    np.testing.assert_allclose(ai, baseline_ai)

    # Multiple turbines with ix filter
    ai = axial_induction_multidim(
        velocities=np.ones((N_TURBINES, 3, 3)) * WIND_CONDITION_BROADCAST,  # 16 x 4 x 3 x 3
        yaw_angle=np.zeros((1, N_TURBINES)),
        tilt_angle=np.ones((1, N_TURBINES)) * 5.0,
        ref_tilt_cp_ct=np.ones((1, N_TURBINES)) * 5.0,
        fCt=np.tile(
            [turbine.fCt_interp[(2, 1)]],
            (
                np.shape(WIND_CONDITION_BROADCAST)[0],
                N_TURBINES,
            )
        ),
        tilt_interp={turbine.turbine_type: None},
        correct_cp_ct_for_tilt=np.array([[False] * N_TURBINES]),
        turbine_type_map=turbine_type_map,
        ix_filter=INDEX_FILTER,
    )

    assert len(ai[0]) == len(INDEX_FILTER)

    # Test the 10 m/s wind speed to use the same baseline as above
    np.testing.assert_allclose(ai[2], baseline_ai)


def test_asdict(sample_inputs_fixture: SampleInputs):

    turbine = Turbine.from_dict(sample_inputs_fixture.turbine)
    dict1 = turbine.as_dict()

    new_turb = Turbine.from_dict(dict1)
    dict2 = new_turb.as_dict()

    assert dict1 == dict2
