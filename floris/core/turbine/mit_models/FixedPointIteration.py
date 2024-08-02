from dataclasses import dataclass
from typing import Any, Callable, List, Protocol, Tuple

import numpy as np
from numpy.typing import ArrayLike


class FixedPointIterationCompatible(Protocol):
    def residual(self, *args, **kwargs) -> Tuple[ArrayLike]: ...

    def initial_guess(self, *args, **kwargs) -> Tuple[ArrayLike]: ...


@dataclass
class FixedPointIterationResult:
    converged: bool
    niter: int
    relax: float
    max_resid: float
    x: ArrayLike


def _fixedpointiteration(
    f: Callable[[ArrayLike, Any], np.ndarray],
    x0: np.ndarray,
    args=(),
    kwargs={},
    eps=0.00001,
    maxiter=100,
    relax=0,
    callback=None,
) -> FixedPointIterationResult:
    """
    Performs fixed-point iteration on function f until residuals converge or max
    iterations is reached.

    Args:
        f (Callable): residual function of form f(x, *args, **kwargs) -> np.ndarray
        x0 (np.ndarray): Initial guess
        args (tuple): arguments to pass to residual function. Defaults to ().
        kwargs (dict): keyword arguments to pass to residual function. Defaults to {}.
        eps (float): Convergence tolerance. Defaults to 0.000001.
        maxiter (int): Maximum number of iterations. Defaults to 100.
        callback (Callable): optional callback function at each iteration of the form f(x0) -> None

    Returns:
        FixedPointIterationResult: Solution to residual function.
    """

    for c in range(maxiter):
        residuals = f(x0, *args, **kwargs)

        x0 = [_x0 + (1 - relax) * _r for _x0, _r in zip(x0, residuals)]
        max_resid = [np.nanmax(np.abs(_r)) for _r in residuals]

        if callback:
            callback(x0)

        if all(_r < eps for _r in max_resid):
            converged = True
            break
    else:
        converged = False

    if maxiter == 0:
        return FixedPointIterationResult(False, 0, np.nan, np.nan, x0)
    return FixedPointIterationResult(converged, c, relax, max_resid, x0)


def fixedpointiteration(
    max_iter: int = 100, tolerance: float = 1e-6, relaxation: float = 0.0
) -> FixedPointIterationCompatible:
    """
    Class decorator which adds a __call__ method to the class which performs
    fixed-point iteration.

    Args:
        max_iter (int): Maximum number of iterations (default: 100)
        tolerance (float): Convergence criteria (default: 1e-6)
        relaxation (float): Relaxation factor between 0 and 1 (default: 0.0)

    The class must contain 2 mandatory methods and 3
    optional method:

    mandatory:
    initial_guess(self, *args, **kwargs)
    residual(self, x, *args, **kwargs)

    optional:
    pre_process(self, *args, **kwargs) # Optional
    post_process(self, result:FixedPointIterationResult) # Optional
    callback(self, x) # Optional

    """

    def decorator(cls: FixedPointIterationCompatible) -> Callable:
        def call(self, *args, **kwargs):
            if hasattr(self, "pre_process"):
                self.pre_process(*args, **kwargs)

            callback = self.callback if hasattr(self, "callback") else None

            x0 = self.initial_guess(*args, **kwargs)
            result = _fixedpointiteration(
                self.residual,
                x0,
                args=args,
                kwargs=kwargs,
                eps=tolerance,
                maxiter=max_iter,
                relax=relaxation,
                callback=callback,
            )

            if hasattr(self, "post_process"):
                return self.post_process(result, *args, **kwargs)
            else:
                return result

        setattr(cls, "__call__", call)
        return cls

    return decorator


def adaptivefixedpointiteration(
    max_iter: int = 100, tolerance: float = 1e-6, relaxations: List[float] = [0.0]
) -> Callable:
    """
    Class decorator which adds a __call__ method to the class which performs
    fixed-point iteration. Same as `fixedpointiteration`, but takes a list of
    relaxation factors, and iterates over all of them in order until convergence
    is reached.
    """

    def decorator(cls: FixedPointIterationCompatible) -> Callable:
        def call(self, *args, **kwargs):
            if hasattr(self, "pre_process"):
                self.pre_process(*args, **kwargs)
            callback = self.callback if hasattr(self, "callback") else None

            for relaxation in relaxations:
                x0 = self.initial_guess(*args, **kwargs)
                result = _fixedpointiteration(
                    self.residual,
                    x0,
                    args=args,
                    kwargs=kwargs,
                    eps=tolerance,
                    maxiter=max_iter,
                    relax=relaxation,
                    callback=callback,
                )
                if result.converged:
                    break

            if hasattr(self, "post_process"):
                return self.post_process(result, *args, **kwargs)
            else:
                return result

        setattr(cls, "__call__", call)
        return cls

    return decorator
