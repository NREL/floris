
import pytest
from attr import define, field
from attrs.exceptions import FrozenAttributeError

from floris.core import BaseClass, BaseModel


@define
class ClassTest(BaseModel):
    x: int = field(default=1, converter=int)
    a_string: str = field(default="abc", converter=str)

    def prepare_function() -> dict:
        return {}

    def function() -> None:
        return None


def test_get_model_values():
    """
    BaseClass and therefore BaseModel previously had a method `get_model_values` that
    returned the values of the model parameters. This was removed because it was redundant
    but this test was refactored to test the as_dict method from FromDictMixin.
    This tests that the parameters are changed when set by the user.
    """
    cls = ClassTest(x=4, a_string="xyz")
    values = cls.as_dict()
    assert len(values) == 2
    assert values["x"] == 4
    assert values["a_string"] == "xyz"

def test_NUM_EPS():
    cls = ClassTest(x=4, a_string="xyz")
    assert cls.NUM_EPS == 0.001

    with pytest.raises(FrozenAttributeError):
        cls.NUM_EPS = 2
