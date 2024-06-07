from abc import ABCMeta
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from .FixedPointIteration import fixedpointiteration


@dataclass
class MomentumSolution:
    """Stores the results of the Unified Momentum model solution."""

    Ctprime: float
    yaw: float
    an: Union[float, npt.ArrayLike]
    u4: Union[float, npt.ArrayLike]
    v4: Union[float, npt.ArrayLike]
    x0: Union[float, npt.ArrayLike]
    dp: Union[float, npt.ArrayLike]
    dp_NL: Optional[Union[float, npt.ArrayLike]] = 0.0
    niter: Optional[int] = 1
    converged: Optional[bool] = True
    beta: Optional[float] = 0.0

    @property
    def Ct(self):
        """Returns the thrust coefficient Ct."""
        return self.Ctprime * (1 - self.an) ** 2 * np.cos(self.yaw) ** 2

    @property
    def Cp(self):
        """Returns the power coefficient Cp."""
        return self.Ctprime * ((1 - self.an) * np.cos(self.yaw)) ** 3


class MomentumBase(metaclass=ABCMeta):
    pass


class LimitedHeck(MomentumBase):
    """
    Solves the limiting case when v_4 << u_4. (Eq. 2.19, 2.20). Also takes Numpy
    array arguments.
    """

    def __call__(self, Ctprime: float, yaw: float, **kwargs) -> MomentumSolution:
        """
        Args:
            Ctprime (float): Rotor thrust coefficient.
            yaw (float): Rotor yaw angle (radians).

        Returns:
            Tuple[float, float, float]: induction and outlet velocities.
        """

        a = Ctprime * np.cos(yaw) ** 2 / (4 + Ctprime * np.cos(yaw) ** 2)
        u4 = (4 - Ctprime * np.cos(yaw) ** 2) / (4 + Ctprime * np.cos(yaw) ** 2)
        v4 = (
            -(4 * Ctprime * np.sin(yaw) * np.cos(yaw) ** 2)
            / (4 + Ctprime * np.cos(yaw) ** 2) ** 2
        )
        dp = np.zeros_like(a)
        x0 = np.inf * np.ones_like(a)
        return MomentumSolution(Ctprime, yaw, a, u4, v4, x0, dp)


@fixedpointiteration(max_iter=500, tolerance=0.00001, relaxation=0.1)
class Heck(MomentumBase):
    """
    Solves the iterative momentum equation for an actuator disk model.
    """

    def __init__(self, v4_correction: float = 1.0):
        """
        Initialize the HeckModel instance.

        Args:
            v4_correction (float, optional): The premultiplier of v4 in the Heck
            model. A correction factor applied to v4, with a default value of
            1.0, indicating no correction. Lu (2023) suggests an empirical correction
            of 1.5.

        Example:
            >>> model = HeckModel(v4_correction=1.5)
        """
        self.v4_correction = v4_correction

    def initial_guess(self, Ctprime, yaw):
        sol = LimitedHeck()(Ctprime, yaw)
        return sol.an, sol.u4, sol.v4

    def residual(self, x: np.ndarray, Ctprime: float, yaw: float) -> np.ndarray:
        """
        Residual function of yawed-actuator disk model in Eq. 2.15.

        Args:
            x (np.ndarray): (a, u4, v4)
            Ctprime (float): Rotor thrust coefficient.
            yaw (float): Rotor yaw angle (radians).

        Returns:
            np.ndarray: residuals of induction and outlet velocities.
        """

        a, u4, v4 = x
        e_a = 1 - np.sqrt(1 - u4**2 - v4**2) / (np.sqrt(Ctprime) * np.cos(yaw)) - a

        e_u4 = (1 - 0.5 * Ctprime * (1 - a) * np.cos(yaw) ** 2) - u4

        e_v4 = (
            -self.v4_correction
            * 0.25
            * Ctprime
            * (1 - a) ** 2
            * np.sin(yaw)
            * np.cos(yaw) ** 2
            - v4
        )

        return np.array([e_a, e_u4, e_v4])

    def post_process(self, result, Ctprime: float, yaw: float):
        if result.converged:
            a, u4, v4 = result.x
        else:
            a, u4, v4 = np.nan * np.zeros_like([Ctprime, Ctprime, Ctprime])
        dp = np.zeros_like(a)
        x0 = np.inf * np.ones_like(a)
        return MomentumSolution(
            Ctprime,
            yaw,
            a,
            u4,
            v4,
            x0,
            dp,
            niter=result.niter,
            converged=result.converged,
        )

