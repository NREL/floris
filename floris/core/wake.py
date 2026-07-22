
import inspect

import attrs
from attrs import define, field

from floris.core import (
    BaseClass,
    BaseLibrary,
    BaseModel,
)
from floris.core.wake_combination import (
    FLS,
    MAX,
    SOSFS,
)
from floris.core.wake_model import (
    BaseWakeModel,
    CumulativeCurl,
    EmpiricalGauss,
    Gauss,
    JensenJimenez,
    NoneWake,
    TurbOParkGauss,
)


MODEL_MAP = {
    "combination_model": {
        "fls": FLS,
        "max": MAX,
        "sosfs": SOSFS
    },
    "velocity_model": {
        "none": NoneWake,
        "cc": CumulativeCurl,
        "gauss": Gauss,
        "jensen": JensenJimenez,
        "empirical_gauss": EmpiricalGauss,
        "turboparkgauss": TurbOParkGauss,
    },
}

def _wake_model_converter(model, model_parameters):
    # If wake_model is an instantiated class, return it
    if isinstance(model, BaseWakeModel):
        return model.__class__(**model_parameters)

    # If model is a string, instantiate from MODEL_MAP
    elif isinstance(model, str):
        if model == "none":
            return NoneWake()
        elif model not in MODEL_MAP["velocity_model"]:
            valid_models = list(MODEL_MAP["velocity_model"].keys())
            raise ValueError(
                f"Unknown velocity model '{model}'. "
                f"Expected one of {valid_models}."
            )
        else:
            return MODEL_MAP["velocity_model"][model](**model_parameters)

    # Handle dict representation of a wake model
    elif isinstance(model, dict):
        return BaseLibrary.from_dict(model).__class__(**model_parameters)

    # Otherwise, raise an error
    else:
        raise TypeError(
            "model must be a BaseWakeModel subclass, or a valid velocity-model string."
        )

@define
class WakeModelManager(BaseClass):
    """
    WakeModelManager is a container class for the wake velocity, deflection,
    turbulence, and combination models.

    Args:
        wake (:obj:`dict`): The wake's properties input dictionary
            - velocity_model (str): The name of the velocity model to be instantiated.
            - combination_model (str): The name of the combination model to be instantiated.
    """
    model: str | BaseWakeModel = field()
    parameters: dict = field(converter=dict)
    combination_model: str | BaseModel = field(default="sosfs")

    def __attrs_post_init__(self) -> None:

        self.model = _wake_model_converter(self.model, self.parameters)

        if isinstance(self.combination_model, str):
            self.combination_model = MODEL_MAP["combination_model"][self.combination_model]()
        elif isinstance(self.combination_model, BaseModel):
            pass

    def assign_user_defined_wake_model(self, wake_model: BaseWakeModel):
        self.model = wake_model
