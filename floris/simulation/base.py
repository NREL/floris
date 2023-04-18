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

from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Any,
    Dict,
    Final,
)

import attrs

from floris.logging_manager import LoggerBase
from floris.type_dec import FromDictMixin


class State(Enum):
    UNINITIALIZED = 0
    INITIALIZED = 1
    USED = 2


class BaseClass(LoggerBase, FromDictMixin):
    """
    BaseClass object class. This class does the logging and MixIn class inheritance.
    """

    state = State.UNINITIALIZED


    @classmethod
    def get_model_defaults(cls) -> Dict[str, Any]:
        """Produces a dictionary of the keyword arguments and their defaults.

        Returns
        -------
        Dict[str, Any]
            Dictionary of keyword argument: default.
        """
        return {el.name: el.default for el in attrs.fields(cls)}

    def _get_model_dict(self) -> dict:
        """Convenience method that wraps the `attrs.asdict` method. Returns the object's
        parameters as a dictionary.

        Returns
        -------
        dict
            The provided or default, if no input provided, model settings as a dictionary.
        """
        return attrs.asdict(self)


class BaseModel(BaseClass, ABC):
    """
    BaseModel is the generic class for any wake models. It defines the API required to
    create a valid model.
    """

    NUM_EPS: Final[float] = 0.001  # This is a numerical epsilon to prevent divide by zeros

    @abstractmethod
    def prepare_function() -> dict:
        raise NotImplementedError("BaseModel.prepare_function")

    @abstractmethod
    def function() -> None:
        raise NotImplementedError("BaseModel.function")
