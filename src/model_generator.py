"""Defines the BaseModel parent class for all models to be based upon."""

from src.wake_velocity.refactor_curl import Curl
from src.wake_velocity.refactor_jensen import Jensen


MODEL_MAP = {
    # "wake_combination": {"""The Combination Models"""},
    # "wake_deflection": {"""The Deflection Models"""},
    # "wake_turbulence": {"""The Turbulence Models"""},
    "wake_velocity": {"curl": Curl, "jensen": Jensen},
}


def model_creator(simulation_dict: dict) -> dict:
    wake_models = {}
    for wake_model, models in MODEL_MAP.items():
        model_string = simulation_dict[wake_model]["model_string"]
        wake_models[model_string] = models[model_string].from_dict(
            simulation_dict[wake_model]
        )

    return wake_models
