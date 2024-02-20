
import numpy as np


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
