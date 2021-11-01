from typing import List

import attr
import numpy as np

from src.utilities import Vec3, float_attrib, model_attrib, iter_validator
from src.base_class import BaseClass


@attr.s(auto_attribs=True)
class MultiZone(BaseClass):
    """The MultiZone model computes the wake velocity deficit based
    on the original multi-zone FLORIS model. See
    :cite:`mvm-gebraad2014data,mvm-gebraad2016wind` for more details.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: mvm-

    Args:
        me (:py:obj:`List[float]`): A list of three floats that help determine the slope
            of the diameters of the three wake zones (near wake, far wake, mixing zone)
            as a function of downstream distance.
        we (:py:obj:`float`): Scaling parameter used to adjust the wake expansion,
            helping to determine the slope of the diameters of the three wake zones as a
            function of downstream distance, as well as the recovery of the velocity
            deficits in the wake as a function of downstream distance.
        aU (:py:obj:`float`): A float that is a parameter used to determine the
            dependence of the wake velocity deficit decay rate on the rotor yaw angle.
        bU (:py:obj:`float`): A float that is another parameter used to determine the
            dependence of the wake velocity deficit decay rate on the rotor yaw angle.
        mU (:py:obj:`list`): A list of three floats that are parameters used to
            determine the dependence of the wake velocity deficit decay rate for each of
            the three wake zones on the rotor yaw angle.
    """

    me: List[float] = attr.ib(
        default=[-0.5, 0.3, 1.0],
        converter=float,
        validator=iter_validator(list, float),
        kw_only=True,
    )
    we: float = float_attrib(default=0.05)
    aU: float = float_attrib(default=12.0)
    bU: float = float_attrib(default=1.3)
    mU: float = attr.ib(
        default=[0.5, 1.0, 5.5],
        converter=float,
        validator=iter_validator(list, float),
        kw_only=True,
    )
    model_string: str = model_attrib(default="multizone")

    def function(
        self,
        x_locations: np.ndarray,
        y_locations: np.ndarray,
        z_locations: np.ndarray,
        turbine: np.ndarray,  # we'll see here
        turbine_coord: Vec3,
        deflection_field: np.ndarray,
        # flow_filed shall be deprecated for the following
        grid_wind_speed: np.ndarray,  # flow_field.u_initial
    ) -> None:
        pass
