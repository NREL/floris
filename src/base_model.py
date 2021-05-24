"""Defines the BaseModel parent class for all models to be based upon."""
from typing import Any, Dict

import attr

from src.utilities import FromDictMixin
from src.logging_manager import LoggerBase


@attr.s(auto_attribs=True)
class BaseModel(LoggerBase, FromDictMixin):
    """
    BaseModel object class. This class does the logging and MixIn class inheritance so
    that it can't be overlooked in creating new models.
    """

    model_string: str = attr.ib(default=None, kw_only=True)
    # requires_resolution: bool = attr.ib(default=False, kw_only=True, init=False)
    # model_grid_resolution: str = attr.ib(default=None, kw_only=True, init=False)

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
