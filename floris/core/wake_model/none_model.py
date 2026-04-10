import numexpr as ne
import numpy as np
from attrs import (
    define,
    field,
    fields,
)

from floris.core import (
    BaseModel,
    Farm,
    FlowField,
    FlowFieldPlanarGrid,
    PointsGrid,
    TurbineGrid,
)
from floris.core.wake_model import BaseWakeModel
from floris.utilities import cosd, sind


NUM_EPS = fields(BaseModel).NUM_EPS.default

@define
class NoneWake(BaseWakeModel):

    def __attrs_post_init__(self):
        self.logger.warning("The wake model is set to 'none'. Wake modeling disabled.")

    def turbine_solve(
        self,
        farm: Farm,
        flow_field: FlowField,
        grid: TurbineGrid,
    ) -> None:

        # None wake model does not calculate any velocity deficits, so simply set the flow field
        flow_field.u_sorted = flow_field.u_initial_sorted.copy()

    def point_solve(
        self,
        farm: Farm,
        flow_field: FlowField,
        grid: FlowFieldPlanarGrid | PointsGrid,
    ) -> None:


        # Initialize the turbulence intensity field over the entire flow field grid
        n_points = grid.x_sorted.shape[1]
        ambient_turbulence_intensities = flow_field.turbulence_intensities[:, None, None, None]
        ambient_turbulence_intensities = np.repeat(ambient_turbulence_intensities, n_points, axis=1)

        # None wake model; set to final values.
        flow_field.u_sorted = flow_field.u_initial_sorted
        flow_field.turbulence_intensity_field_sorted = ambient_turbulence_intensities.copy()
