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
 

import sys
from floris.floris import Floris
from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph import GlobbingFilter
from pycallgraph.output import GraphvizOutput

config = Config(groups=True)
config.trace_filter = GlobbingFilter(exclude=[
    'pycallgraph.*',
    'json*',
    'codecs*',
    '_bootlocale*',
    '_weakrefset*'
])

graphviz = GraphvizOutput()
graphviz.output_file = 'initialization.png'
with PyCallGraph(config=config, output=graphviz):
    if len(sys.argv) > 1:
        floris = Floris(sys.argv[1])
    else:
        floris = Floris("example_input.json")

graphviz = GraphvizOutput()
graphviz.output_file = 'calculate_wake.png'
with PyCallGraph(config=config, output=graphviz):
    floris.farm.flow_field.calculate_wake()
