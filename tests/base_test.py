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

from floris.simulation import BaseClass, BaseModel


@define
class ClassTest(BaseModel):
    x: int = field(default=1, converter=int)
    a_string: str = field(default="abc", converter=str)

    def prepare_function() -> dict:
        return {}

    def function() -> None:
        return None


def test_get_model_defaults():
    """
    This tests that the default parameter values are set correctly and unchanged if not
    explicitly set.
    Note that the attributes in BaseClass and BaseModel are treated as default parameters
    by attrs.
    """
    defaults = ClassTest.get_model_defaults()
    assert len(defaults) == 5
    assert defaults["x"] == 1
    assert defaults["a_string"] == "abc"


def test_get_model_values():
    """
    This tests that the parameters are changed when set by the user.
    Note that the attributes in BaseClass and BaseModel are treated as parameters by attrs.
    They aren't changed in the instantiation method, but they are still included in the
    parameter set.
    """
    cls = ClassTest(x=4, a_string="xyz")
    values = cls._get_model_dict()
    assert len(values) == 5
    assert values["x"] == 4
    assert values["a_string"] == "xyz"

def test_NUM_EPS():
    cls = ClassTest(x=4, a_string="xyz")
    assert cls.NUM_EPS == 0.001

    with pytest.raises(AssertionError):
        cls.NUM_EPS = 2
