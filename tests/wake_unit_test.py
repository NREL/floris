
import numpy as np

from floris.core import (
    Core,
    power,
    WakeModelManager,
)
from tests.conftest import SampleInputs


def test_asdict(sample_inputs_fixture: SampleInputs):

    wake_model_manager = WakeModelManager.from_dict(sample_inputs_fixture.wake)
    dict1 = wake_model_manager.as_dict()

    new_wake = WakeModelManager.from_dict(dict1)
    dict2 = new_wake.as_dict()

    assert dict1 == dict2

def test_combination_model(sample_inputs_fixture):
    """
    Tandem turbines
    """
    sample_inputs_fixture.switch_wake_model("jensen")

    floris = Core.from_dict(sample_inputs_fixture.core)
    floris.initialize_domain()
    floris.solve_for_turbines()

    velocities = floris.flow_field.u
    turbulence_intensities = floris.flow_field.turbulence_intensity_field
    air_density = floris.flow_field.air_density
    yaw_angles = floris.farm.yaw_angles
    power_setpoints = floris.farm.power_setpoints
    awc_modes = floris.farm.awc_modes
    awc_amplitudes = floris.farm.awc_amplitudes

    farm_powers_sosfs = power(
        turbines=floris.farm.turbines,
        velocities=velocities,
        turbulence_intensities=turbulence_intensities,
        air_density=air_density,
        yaw_angles=yaw_angles,
        power_setpoints=power_setpoints,
        awc_modes=awc_modes,
        awc_amplitudes=awc_amplitudes,
        turbine_type_map=floris.farm.turbine_type_map,
    )

    # Switch to a different combination model and rerun
    sample_inputs_fixture.core["wake"]["combination_model"] = "fls"

    floris = Core.from_dict(sample_inputs_fixture.core)
    floris.initialize_domain()
    floris.solve_for_turbines()

    velocities = floris.flow_field.u
    turbulence_intensities = floris.flow_field.turbulence_intensity_field
    air_density = floris.flow_field.air_density
    yaw_angles = floris.farm.yaw_angles
    power_setpoints = floris.farm.power_setpoints
    awc_modes = floris.farm.awc_modes
    awc_amplitudes = floris.farm.awc_amplitudes

    farm_powers_fls = power(
        turbines=floris.farm.turbines,
        velocities=velocities,
        turbulence_intensities=turbulence_intensities,
        air_density=air_density,
        yaw_angles=yaw_angles,
        power_setpoints=power_setpoints,
        awc_modes=awc_modes,
        awc_amplitudes=awc_amplitudes,
        turbine_type_map=floris.farm.turbine_type_map,
    )

    # First-row turbines should be the same. Downstream turbines should differ some
    assert np.allclose(farm_powers_sosfs[0, 0], farm_powers_fls[0,0])
    assert not np.allclose(farm_powers_sosfs, farm_powers_fls)
