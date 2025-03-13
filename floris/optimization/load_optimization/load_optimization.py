"""Module for the load optimization class and functions."""

from floris import FlorisModel
import numpy as np


def compute_load_ti(fmodel,
                    load_ambient_ti,
                    wake_slope = 0.3,
                    max_dist_D = 10.0):
    """Compute the turbine load and turbulence intensity for the current layout.

    Args:
        fmodel: FlorisModel instance


    """

    # TODO for now assume the first one
    D = fmodel.core.farm.rotor_diameters[0,0]


    # Initialize the load_ti to the load_ambient_ti
    load_ti = np.ones((fmodel.n_findex, fmodel.n_turbines)) * load_ambient_ti