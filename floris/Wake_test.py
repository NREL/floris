"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from .Wake import Wake
from .WakeDeflection import WakeDeflection
import numpy as np
import pytest

def test_instatiation():
    """
    # well this really needs to be made better
    """
    dictionary = {
        "description": "test",
        "properties": {
            "velocity_model": "floris",
            "deflection_model": "gauss",
            "parameters": {
                "jensen": {
                    "we": 0.05
                },
                "floris": {
                    "me": [
                        -0.05,
                        0.3,
                        1.0
                    ],
                    "aU": 12.0,
                    "bU": 1.3,
                    "mU": [
                        0.5,
                        1.0,
                        5.5
                    ]
                },
                "gauss": {
                    "ka": 0.3,
                    "kb": 0.004,
                    "alpha": 0.58,
                    "beta": 0.077
                },
                "jimenez": {
                    "kd": 0.17,
                    "ad": 0.0,
                    "bd": 0.0
                },
                "gauss_deflection": {
                    "ka": 0.3,
                    "kb": 0.004,
                    "alpha": 0.58,
                    "beta": 0.077,
                    "ad": 0.0,
                    "bd": 0.0
                }
            }
        }
    }
    assert Wake(dictionary) != None
