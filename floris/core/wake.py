
import attrs
from attrs import define, field

from floris.core import (
    BaseClass,
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
    model: BaseWakeModel = field(init=False)
    user_defined_wake_model: BaseWakeModel | None = field(default=None)

    def __attrs_post_init__(self) -> None:
        velocity_model_string = self.model_strings["velocity_model"].lower()
        if velocity_model_string == "none":
            self.model = NoneWake()
        else:
            model_parameters = self._temp_create_single_wake_model_dict()
            self.model = MODEL_MAP["velocity_model"][velocity_model_string](**model_parameters)

        combination_model_string = self.model_strings["combination_model"].lower()
        combination_model: BaseModel = MODEL_MAP["combination_model"][combination_model_string]
        self.combination_model = combination_model()

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

    def _temp_create_single_wake_model_dict(self):
        """
        This is a temporary function until wake model parametrization is unified on the
        input dictionary. However, we may use it going forward for back compatibility with
        v4. In that case, checks should be made to ensure compatible deficit/deflection/turbulence
        models are being used together.
        """
        vel_model = self.model_strings["velocity_model"].lower()
        if vel_model == "gauss":
            model_parameters = self.wake_velocity_parameters["gauss"] | \
                self.wake_deflection_parameters["gauss"] | \
                self.wake_turbulence_parameters["crespo_hernandez"]
            model_parameters["enable_transverse_velocities"] = self.enable_transverse_velocities
            model_parameters["enable_yaw_added_recovery"] = self.enable_yaw_added_recovery
            model_parameters["enable_secondary_steering"] = self.enable_secondary_steering
        elif vel_model == "cc":
            model_parameters = self.wake_velocity_parameters["cc"] | \
                self.wake_deflection_parameters["gauss"] | \
                self.wake_turbulence_parameters["crespo_hernandez"]
            model_parameters["enable_transverse_velocities"] = self.enable_transverse_velocities
            model_parameters["enable_yaw_added_recovery"] = self.enable_yaw_added_recovery
            model_parameters["enable_secondary_steering"] = self.enable_secondary_steering
        elif vel_model == "jensen":
            model_parameters = self.wake_velocity_parameters["jensen"] | \
                self.wake_deflection_parameters["jimenez"] | \
                self.wake_turbulence_parameters["crespo_hernandez"]
        elif vel_model == "turboparkgauss":
            model_parameters = self.wake_velocity_parameters["turboparkgauss"]
        elif vel_model == "empirical_gauss":
            model_parameters = self.wake_velocity_parameters["empirical_gauss"] | \
                self.wake_deflection_parameters["empirical_gauss"] | \
                self.wake_turbulence_parameters["wake_induced_mixing"]
            model_parameters["enable_yaw_added_recovery"] = self.enable_yaw_added_recovery
            model_parameters["enable_active_wake_mixing"] = self.enable_active_wake_mixing
        elif vel_model == "none":
            model_parameters = {}
        else:
            model_parameters = {}

        return model_parameters
