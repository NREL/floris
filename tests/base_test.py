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
from attr import define, field
from attrs.exceptions import FrozenAttributeError

from floris.simulation import BaseClass, BaseModel


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
