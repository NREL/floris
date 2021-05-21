"""Defines the BaseModel parent class for all models to be based upon."""
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
    requires_resolution: bool = attr.ib(default=False, kw_only=True)
    model_grid_resolution: str = attr.ib(default=None, kw_only=True)

    def __attrs_post_init__(self) -> None:
        if self.model_string is None:
            raise ValueError("No 'model_string' defined.")
