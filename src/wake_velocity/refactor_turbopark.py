import attr
import numpy as np

from src.utilities import Vec3, is_default
from src.base_model import BaseModel


@attr.s(auto_attribs=True)
class TurbOPark(BaseModel):
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

    A: float = attr.ib(default=0.06, converter=float, kw_only=True)
    c1: float = attr.ib(default=1.5, converter=float, kw_only=True)
    c2: float = attr.ib(default=0.08, converter=float, kw_only=True)
    model_string: str = attr.ib(
        default="turbopark", on_setattr=attr.setters.frozen, validator=is_default
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
        turbulence_intensity: np.ndarray,  # flow_field.turbulence_intensity
    ) -> None:
        pass
