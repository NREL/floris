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


import re
import sys
import time
import cProfile
from copy import deepcopy

from floris.simulation import Floris


# Provide a path to an input file
# floris = Floris.from_json(sys.argv[1])

# Or use a default. If using default, this script must be called from the root dir
# i.e. cd floris/ && python tests/profiling.py
floris = Floris.from_json("examples/example_input.json")

floris.steady_state_atmospheric_condition()

def run_floris():
    floris = Floris.from_json("examples/example_input.json")
    return floris

start = time.time()
cProfile.run('re.compile("floris.steady_state_atmospheric_condition()")')
end = time.time()

print(start, end, end - start)
