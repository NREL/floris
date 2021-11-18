import attr
import numpy as np

from src.utilities import Vec3, float_attrib, model_attrib
from src.base_class import BaseClass


@attr.s(auto_attribs=True)
class TurbOPark(BaseClass):
    """An implementation of the TurbOPark model by Nicolai Nygaard
    :cite:`jvm-nygaard2020modelling`.
    Default tuning calibrations taken from same paper.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: jvm-

    Args:
        A (:py:obj:`float`): ???
        c1 (:py:obj:`float`): ???
        c2 (:py:obj:`float`): ???
    """

    A: float = float_attrib(default=0.06)
    c1: float = float_attrib(default=1.5)
    c2: float = float_attrib(default=0.08)
    model_string: str = model_attrib(default="turbopark")

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
        turbulence_intensity: np.ndarray,  # flow_field.turbulence_intensity
    ) -> None:
        pass
