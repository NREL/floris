from typing import Any, Dict, List, Union
from itertools import product

import attr
import numpy as np

from src.utilities import Vec3, is_default
from src.base_model import BaseModel


@attr.s(auto_attribs=True)
class Jensen(BaseModel):
    """The Jensen model computes the wake velocity deficit based on the classic
    Jensen/Park model :cite:`jvm-jensen1983note`.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: jvm-

    Args:
        we (:py:obj:`float`): The linear wake decay constant that defines the cone
            boundary for the wake as well as the velocity deficit. D/2 +/- we*x is the
            cone boundary for the wake.
    """

    we: float = attr.ib(default=-0.05, converter=float, kw_only=True)
    model_string: str = attr.ib(
        default="jensen", on_setattr=attr.setters.frozen, validator=is_default
    )

    def function(
        self,
        x_locations: np.ndarray,
        y_locations: np.ndarray,
        z_locations: np.ndarray,
        turbine: np.ndarray,  # we'll see here
        turbine_coord: Vec3,
        deflection_field: np.ndarray,
        # flow_filed shall be deprecated for the following
        u_initial: np.ndarray,  # flow_field.u_initial
    ) -> None:
        pass
