from floris.core import Core
from floris.core.wake_turbulence import NoneWakeTurbulence


VELOCITY_MODEL = "gauss"
DEFLECTION_MODEL = "gauss"

def test_NoneWakeTurbulence(sample_inputs_fixture):

    turbulence_intensities = [0.1, 0.05]

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["turbulence_model"] = "none"
    sample_inputs_fixture.core["farm"]["layout_x"] = [0.0, 0.0, 600.0, 600.0]
    sample_inputs_fixture.core["farm"]["layout_y"] = [0.0, 600.0, 0.0, 600.0]
    sample_inputs_fixture.core["flow_field"]["wind_directions"] = [270.0, 360.0]
    sample_inputs_fixture.core["flow_field"]["wind_speeds"] = [8.0, 8.0]
    sample_inputs_fixture.core["flow_field"]["turbulence_intensities"] = turbulence_intensities

    core = Core.from_dict(sample_inputs_fixture.core)
    core.initialize_domain()
    core.steady_state_atmospheric_condition()

    assert (
        core.flow_field.turbulence_intensity_field_sorted[0,:] == turbulence_intensities[0]
    ).all()
    assert (
        core.flow_field.turbulence_intensity_field_sorted[1,:] == turbulence_intensities[1]
    ).all()
