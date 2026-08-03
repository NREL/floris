"""
Library of functions defining wake combination approaches.
"""

import numpy as np


def sosfs(wake_field: np.ndarray, velocity_field: np.ndarray):
    """
    Combines the base flow field with the velocity deficits
    using sum of squares.

    Args:
        u_field (np.array): The base flow field.
        u_wake (np.array): The wake to apply to the base flow field.

    Returns:
        np.array: The resulting flow field after applying the wake to the
            base.
    """
    return np.hypot(wake_field, velocity_field)


def maximum(wake_field: np.ndarray, velocity_field: np.ndarray):
    """
    Incorporates the velocity deficits into the base flow field by
    selecting the maximum of the two for each point.

    Args:
        u_field (np.array): The base flow field.
        u_wake (np.array): The wake to apply to the base flow field.

    Returns:
        np.array: The resulting flow field after applying the wake to the
            base.
    """
    return np.maximum(wake_field, velocity_field)


def fls(wake_field: np.ndarray, velocity_field: np.ndarray):
    """
    Combines the base flow field with the velocity deficits
    using freestream linear superposition. In other words, the wake
    field and base fields are simply added together.

    Args:
        u_field (np.array): The base flow field.
        u_wake (np.array): The wake to apply to the base flow field.

    Returns:
        np.array: The resulting flow field after applying the wake to the
            base.
    """
    return wake_field + velocity_field

def none_combination(wake_field: np.ndarray, velocity_field: np.ndarray):
    """
    Return None, indicating no combination is applied. Likely will not be called
    in a functional model.

    Args:
        wake_field (np.array): The wake to apply to the base flow field.
        velocity_field (np.array): The base flow field.

    Returns:
        None
    """
    return None
