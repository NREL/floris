# Copyright 2020 NREL

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

# Default Class
# The default class uses the legacy gauss velocity deficit model, however the
# Crespo-Hernandez Turbulence Model has been retuned to match available data
# GCH is used for deflection in place of the deflection model
fi_1 = wfct.floris_interface.FlorisInterface("../example_input.json")
fi_dict["default"] = (fi_1, "b", "d")  # Define a fixed color and marker

# Merged Class
# The merged case, still in development, blends the super-gaussian model of
#       Blondel, F. and Cathelain, M. "An alternative form of the
#       super-Gaussian wind turbine wake model."
#       *Wind Energy Science Disucssions*,   2020.
# Into the gaussian model to improve near wake matching while attempting to hold
# far wake values consistent with previous resaults
# The TI model is as in the default case, and GCH is enabled
fi_2 = wfct.floris_interface.FlorisInterface("../other_jsons/input_merge.json")
fi_dict["merge"] = (fi_2, "r", "o")

# Legacy version
# The legacy model uses the settings which were default in the previous version
# of FLORIS, including the legacy gaussian model, TI settings, and GCH disabled,
# With a deflection multiplier of 1.2
fi_3 = wfct.floris_interface.FlorisInterface("../other_jsons/input_legacy.json")
fi_dict["legacy"] = (fi_3, "m", "^")

pickle.dump(fi_dict, open("floris_models.p", "wb"))
