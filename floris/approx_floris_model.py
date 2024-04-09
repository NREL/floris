from __future__ import annotations

from pathlib import Path

from floris import UncertainFlorisModel


class ApproxFlorisModel(UncertainFlorisModel):
    """
    The ApproxFlorisModel overloads the UncertainFlorisModel with the special case that
    the wd_sample_points = [0].  This is a special case where no uncertainty is added
    but the resolution of the values wind direction, wind speed etc are still reduced
    by the specified resolution.  This allows for cases to be reused and a faster approximate
    result computed
    """

    def __init__(
        self,
        configuration: dict | str | Path,
        wd_resolution=1.0,  # Degree
        ws_resolution=1.0,  # m/s
        ti_resolution=0.01,
        yaw_resolution=1.0,  # Degree
        power_setpoint_resolution=100,  # kW
        verbose=False,
    ):
        super().__init__(
            configuration,
            wd_resolution,
            ws_resolution,
            ti_resolution,
            yaw_resolution,
            power_setpoint_resolution,
            wd_std=1.0,
            wd_sample_points=[0],
            fix_yaw_to_nominal_direction=False,
            verbose=verbose,
        )

        self.wd_resolution = wd_resolution
        self.ws_resolution = ws_resolution
        self.ti_resolution = ti_resolution
        self.yaw_resolution = yaw_resolution
        self.power_setpoint_resolution = power_setpoint_resolution
