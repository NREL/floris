from pathlib import Path

from floris.tools.floris_interface import FlorisInterface


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full_v3.yaml"
JSON_INPUT = TEST_DATA / "input_full_v3.json"


def test_read_yaml():
    fi = FlorisInterface(configuration=YAML_INPUT)
    assert isinstance(fi, FlorisInterface)


def test_calculate_wake():
    pass


def test_reinitialize_flow_field():
    pass


def test_get_plane_of_points():
    pass


def test_get_set_of_points():
    pass


def test_get_hor_plane():
    pass


def test_get_cross_plane():
    pass


def test_get_y_plane():
    pass


def test_get_flow_data():
    pass


def test_get_farm_power():
    pass


def test_get_turbine_layout():
    pass


def test_get_power_curve():
    pass


def test_get_turbine_ti():
    pass


def test_get_farm_power_for_yaw_angle():
    pass


def test_get_farm_AEP():
    pass


def test_calc_one_AEP_case():
    pass


def test_get_farm_AEP_parallel():
    pass


def test_calc_AEP_wind_limit():
    pass


def test_calc_change_turbine():
    pass


def test_set_use_points_on_perimeter():
    pass


def test_set_gch():
    pass


def test_set_gch_yaw_added_recovery():
    pass


def test_set_gch_secondary_steering():
    pass


def test_layout_x():  # TODO
    pass


def test_layout_y():  # TODO
    pass


def test_TKE_to_TI():
    pass


def test_set_rotor_diameter():  # TODO
    pass


def test_show_model_parameters():  # TODO
    pass


def test_get_model_parameters():  # TODO
    pass


def test_set_model_parameters():  # TODO
    pass


def test_vis_layout():
    pass


def test_show_flow_field():
    pass
