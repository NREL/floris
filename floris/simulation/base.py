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


"""
Defines the BaseClass parent class for all models to be based upon.
"""

from abc import abstractmethod
from enum import Enum
from typing import (
    Any,
    Dict,
    Final,
)

from attrs import (
    Attribute,
    define,
    field,
    fields,
    setters,
)

from floris.logging_manager import LoggingManager
from floris.type_dec import FromDictMixin


class State(Enum):
    UNINITIALIZED = 0
    INITIALIZED = 1
    USED = 2


@define
class BaseClass(FromDictMixin):
    """
    BaseClass object class. This class does the logging and MixIn class inheritance.
    """

    # Initialize `state` and ensure it is treated as an attribute rather than a constant parameter.
    # See https://www.attrs.org/en/stable/api-attr.html#attr.ib
    state = field(init=False, default=State.UNINITIALIZED)
    _logging_manager: LoggingManager = field(init=False, default=LoggingManager())

    @property
    def logger(self):
        """Returns the logger manager object."""
        return self._logging_manager.logger

@define
class BaseModel(BaseClass):
    """
    BaseModel is the generic class for any wake models. It defines the API required to
    create a valid model.
    """

    # This is a numerical epsilon to prevent divide by zeros
    NUM_EPS: Final[float] = field(init=False, default=0.001, on_setattr=setters.frozen)

    @abstractmethod
    def prepare_function() -> dict:
        raise NotImplementedError("BaseModel.prepare_function")

    @abstractmethod
    def function() -> None:
        raise NotImplementedError("BaseModel.function")
