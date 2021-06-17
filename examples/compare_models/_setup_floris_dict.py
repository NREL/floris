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


# Setup a dictionary of FLORIS cases to use consistently across these examples

import copy
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import floris.tools as wfct


# Declare dictionary
fi_dict = dict()

# Jensen
fi_3 = wfct.floris_interface.FlorisInterface("../other_jsons/jensen.json")
fi_3.reinitialize_flow_field(
    wind_speed=[8.0], wind_direction=[270.0], turbulence_intensity=[0.06]
)
fi_dict["jensen"] = (fi_3, "k", "*")  # Define a fixed color and marker

# Turbo Park
fi_2 = wfct.floris_interface.FlorisInterface("../other_jsons/turbopark.json")
fi_2.reinitialize_flow_field(
    wind_speed=[8.0], wind_direction=[270.0], turbulence_intensity=[0.06]
)
fi_dict["turbo"] = (fi_2, "r", "s")  # Define a fixed color and marker

# Default Class
# Gaussian model with near wake however the
# Crespo-Hernandez Turbulence Model has been retuned to match available data
# GCH is used for deflection in place of the deflection model
fi_1 = wfct.floris_interface.FlorisInterface("../example_input.json")
fi_1.reinitialize_flow_field(
    wind_speed=[8.0], wind_direction=[270.0], turbulence_intensity=[0.06]
)
fi_dict["default"] = (fi_1, "b", "d")  # Define a fixed color and marker

pickle.dump(fi_dict, open("floris_models.p", "wb"))
