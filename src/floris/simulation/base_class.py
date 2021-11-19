"""Defines the BaseClass parent class for all models to be based upon."""
from abc import abstractmethod, abstractstaticmethod
from typing import Any, Dict

import attr
from floris.utilities import FromDictMixin
from floris.logging_manager import LoggerBase


@attr.s(auto_attribs=True)
class BaseClass(LoggerBase, FromDictMixin):
    """
    BaseClass object class. This class does the logging and MixIn class inheritance so
    that it can't be overlooked in creating new models.
    """

    model_string: str = attr.ib(default=None, kw_only=True)

    def __attrs_post_init__(self) -> None:
        if self.model_string is None:
            raise ValueError("No 'model_string' defined.")

    @classmethod
    def get_model_defaults(cls) -> Dict[str, Any]:
        """Produces a dictionary of the keyword arguments and their defaults.

        Returns
        -------
        Dict[str, Any]
            Dictionary of keyword argument: default.
        """
        return {el.name: el.default for el in attr.fields(cls)}

    def _get_model_dict(self) -> dict:
        """Convenience method that wraps the `attr.asdict` method. Returns the model's
        parameters as a dictionary.

        Returns
        -------
        dict
            The provided or default, if no input provided, model settings as a dictionary.
        """
        return attr.asdict(self)

    @abstractstaticmethod
    def prepare_function() -> None:
        """This method must be implemented in every created model. It should take a
        `TurbineGrid`, `Farm`, and `FlowField` object and return a dictionary of the
        the required parameters for `function()`.
        """
        pass

    @abstractmethod
    def function() -> None:
        """The actual model for the wake deflection, wake velocity, wake combination,
        or wake turbulence model being created.
        """
        pass
