
from floris.core import WakeModelManager
from tests.conftest import SampleInputs


def test_asdict(sample_inputs_fixture: SampleInputs):

    wake_model_manager = WakeModelManager.from_dict(sample_inputs_fixture.wake)
    dict1 = wake_model_manager.as_dict()

    new_wake = WakeModelManager.from_dict(dict1)
    dict2 = new_wake.as_dict()

    assert dict1 == dict2
