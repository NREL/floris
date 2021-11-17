from pathlib import Path

from src.tools.floris_interface import FlorisInterface


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full_v3.yaml"
JSON_INPUT = TEST_DATA / "input_full_v3.json"


def test_read_json():
    fi = FlorisInterface(configuration=JSON_INPUT)
    assert isinstance(fi, FlorisInterface)


def test_read_yaml():
    print(YAML_INPUT)
    fi = FlorisInterface(configuration=YAML_INPUT)
    assert isinstance(fi, FlorisInterface)
