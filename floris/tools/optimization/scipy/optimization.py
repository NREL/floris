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

import numpy as np
import matplotlib.pyplot as plt


try:
    from mpi4py.futures import MPIPoolExecutor
except ImportError:
    pass


class Optimization:
    """
    Optimization is the base optimization class for
    `~.tools.optimization.scipy` subclasses. Contains some common
    methods and properties that can be used by the individual optimization
    classes.
    """

    def __init__(self, fi):
        """
        Initializes an Optimization object by assigning a
        FlorisInterface object.

        Args:
            fi (:py:class:`~.tools.floris_interface.FlorisInterface`):
                Interface used to interact with the Floris object.
        """
        self.fi = fi

    # Private methods

    def _reinitialize(self):
        pass

    def _norm(self, val, x1, x2):
        return (val - x1) / (x2 - x1)

    def _unnorm(self, val, x1, x2):
        return np.array(val) * (x2 - x1) + x1

    # Properties

    @property
    def nturbs(self):
        """
        Number of turbines in the :py:class:`~.farm.Farm` object.

        Returns:
            int
        """
        self._nturbs = len(self.fi.floris.farm.turbine_map.turbines)
        return self._nturbs
