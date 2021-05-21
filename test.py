from src.model_generator import model_creator


if __name__ == "__main__":

    model_dictionary_basic_curl = {"wake_velocity": {"model_string": "curl"}}

    model_dictionary_basic_jensen = {"wake_velocity": {"model_string": "jensen"}}

    model_dictionary_fully_defined_curl = {
        "wake_velocity": {
            "model_grid_resolution": [250, 100, 75],
            "initial_deficit": 2.0,
            "dissipation": 0.06,
            "veer_linear": 0.0,
            "initial": 0.1,
            "constant": 0.73,
            "ai": 0.8,
            "downstream": -0.275,
            "model_string": "curl",
        }
    }

    print(model_creator(model_dictionary_basic_curl))

    print(model_creator(model_dictionary_basic_jensen))

    print(model_creator(model_dictionary_fully_defined_curl))
