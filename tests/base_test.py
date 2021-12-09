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


import attr
import pytest

from floris.simulation import BaseClass, BaseModel


@attr.s(auto_attribs=True)
class ClassTest(BaseClass):
    x: int = attr.ib(default=1, converter=int)
    model_string: str = attr.ib(default="test", converter=str)


def test_get_model_defaults():
    defaults = ClassTest.get_model_defaults()
    assert len(defaults) == 2
    assert defaults["x"] == 1
    assert defaults["model_string"] == "test"


def test_get_model_values():
    cls = ClassTest(x=4, model_string="new")
    values = cls._get_model_dict()
    assert len(values) == 2
    assert values["x"] == 4
    assert values["model_string"] == "new"
