
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


@define
class WakeModelManager(BaseClass):
    # TODO: Will likely want to reconfigure this, eventually
    """
    WakeModelManager is a container class for the wake velocity, deflection,
    turbulence, and combination models.

    Args:
        wake (:obj:`dict`): The wake's properties input dictionary
            - velocity_model (str): The name of the velocity model to be instantiated.
            - combination_model (str): The name of the combination model to be instantiated.
    """
    model_strings: dict = field(converter=dict)
    enable_secondary_steering: bool = field(converter=bool)
    enable_yaw_added_recovery: bool = field(converter=bool)
    enable_active_wake_mixing: bool = field(converter=bool)
    enable_transverse_velocities: bool = field(converter=bool)

    wake_deflection_parameters: dict = field(converter=dict)
    wake_turbulence_parameters: dict = field(converter=dict)
    wake_velocity_parameters: dict = field(converter=dict, factory=dict)

    # TODO: How should I handle combination models going forward?
    combination_model: BaseModel = field(init=False)
    velocity_model: BaseLibrary = field(init=False)
    user_defined_wake_model: BaseWakeModel | None = field(default=None)

    def __attrs_post_init__(self) -> None:
        velocity_model_string = self.model_strings["velocity_model"].lower()
        if velocity_model_string == "none":
            self.velocity_model = NoneWake()
        else:
            self.velocity_model = MODEL_MAP["velocity_model"][velocity_model_string](
                self.wake_velocity_parameters[velocity_model_string]
            )

        combination_model_string = self.model_strings["combination_model"].lower()
        model: BaseModel = MODEL_MAP["combination_model"][combination_model_string]
        self.combination_model = model()

    def assign_user_defined_wake_model(self, wake_model: BaseWakeModel):
        self.user_defined_wake_model = wake_model

    @model_strings.validator
    def validate_model_strings(self, instance: attrs.Attribute, value: dict) -> None:
        required_strings = [
            "velocity_model",
            "deflection_model",
            "combination_model",
            "turbulence_model"
        ]
        # Check that all required strings are given
        for s in required_strings:
            if s not in value.keys():
                raise KeyError(f"Wake: '{s}' not provided in the input but it is required.")

        # Check that no other strings are given
        for k in value.keys():
            if k not in required_strings:
                raise KeyError((
                    f"Wake: '{k}' was given as input but it is not a valid option."
                    f"Required inputs are: {', '.join(required_strings)}"
                ))

    @property
    def deflection_function(self):
        return self.deflection_model.function

    @property
    def velocity_function(self):
        return self.velocity_model.function

    @property
    def turbulence_function(self):
        return self.turbulence_model.function

    @property
    def combination_function(self):
        return self.combination_model.function
