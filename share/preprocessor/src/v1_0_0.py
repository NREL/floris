
from .version_class import VersionClass
from .data_transform import DataTransform


class V1_0_0(VersionClass, DataTransform):

    version_string = "v1.0.0"

    default_meta = {
        "type": "floris_input",
        "name": "floris_input_file",
        "description": "FLORIS input file",
        "version": version_string
    }

    default_farm = {
        "type": "farm",
        "name": "farm_example",
        "description": "Example 2x2 Wind Farm",
        "properties": {
            "wind_speed": 8.0,
            "wind_direction": 270.0,
            "turbulence_intensity": 0.06,
            "wind_shear": 0.12,
            "wind_veer": 0.0,
            "air_density": 1.225,
            "layout_x": [
                0.0,
                800.0,
                0.0,
                800.0
            ],
            "layout_y": [
                0.0,
                0.0,
                630.0,
                630.0
            ]
        }
    }

    default_wake = {
        "name": "wake_default",
        "description": "wake",
        "properties": {
            "velocity_model": "gauss",
            "deflection_model": "gauss",
            "combination_model": "sosfs",
            "parameters": {
                "turbulence_intensity": {
                    "initial": 0.1,
                    "constant": 0.73,
                    "ai": 0.8,
                    "downstream": -0.275
                },
                "jensen": {
                    "we": 0.05
                },
                "multizone": {
                    "me": [
                        -0.5,
                        0.3,
                        1.0
                    ],
                    "we": 0.05,
                    "aU": 12.0,
                    "bU": 1.3,
                    "mU": [
                        0.5,
                        1.0,
                        5.5
                    ]
                },
                "gauss": {
                    "ka": 0.3,
                    "kb": 0.004,
                    "alpha": 0.58,
                    "beta": 0.077,
                    "ad": 0.0,
                    "bd": 0.0
                },
                "jimenez": {
                    "kd": 0.05,
                    "ad": 0.0,
                    "bd": 0.0
                },
                "curl": {
                    "model_grid_resolution": [
                        250,
                        100,
                        75
                    ],
                    "initial_deficit": 2.0,
                    "dissipation": 0.06,
                    "veer_linear": 0.0
                }
            }
        }
    }

    default_turbine = {
        "type": "turbine",
        "name": "nrel_5mw",
        "description": "NREL 5MW",
        "properties": {
            "rotor_diameter": 126.0,
            "hub_height": 90.0,
            "blade_count": 3,
            "pP": 1.88,
            "pT": 1.88,
            "generator_efficiency": 1.0,
            "power_thrust_table": {
                "power": [0.0, 0.0,  0.1780851 ,  0.28907459,  0.34902166,
                    0.3847278 ,  0.40605878,  0.4202279 ,  0.42882274,  0.43387274,
                    0.43622267,  0.43684468,  0.43657497,  0.43651053,  0.4365612 ,
                    0.43651728,  0.43590309,  0.43467276,  0.43322955,  0.43003137,
                    0.37655587,  0.33328466,  0.29700574,  0.26420779,  0.23839379,
                    0.21459275,  0.19382354,  0.1756635 ,  0.15970926,  0.14561785,
                    0.13287856,  0.12130194,  0.11219941,  0.10311631,  0.09545392,
                    0.08813781,  0.08186763,  0.07585005,  0.07071926,  0.06557558,
                    0.06148104,  0.05755207,  0.05413366,  0.05097969,  0.04806545,
                    0.04536883,  0.04287006,  0.04055141
                ],
                "thrust": [1.19187945, 1.17284634, 1.09860817, 1.02889592, 0.97373036,
                    0.92826162, 0.89210543, 0.86100905, 0.835423  , 0.81237673,
                    0.79225789, 0.77584769, 0.7629228 , 0.76156073, 0.76261984,
                    0.76169723, 0.75232027, 0.74026851, 0.72987175, 0.70701647,
                    0.54054532, 0.45509459, 0.39343381, 0.34250785, 0.30487242,
                    0.27164979, 0.24361964, 0.21973831, 0.19918151, 0.18131868,
                    0.16537679, 0.15103727, 0.13998636, 0.1289037 , 0.11970413,
                    0.11087113, 0.10339901, 0.09617888, 0.09009926, 0.08395078,
                    0.0791188 , 0.07448356, 0.07050731, 0.06684119, 0.06345518,
                    0.06032267, 0.05741999, 0.05472609
                ],
                "wind_speed": [ 2.0 ,  2.5,  3.0 ,  3.5,  4.0 ,  4.5,  5.0 ,  5.5,  6.0 ,  6.5,  7.0 ,
                    7.5,  8.0 ,  8.5,  9.0 ,  9.5, 10.0 , 10.5, 11.0 , 11.5, 12.0 , 12.5,
                    13.0 , 13.5, 14.0 , 14.5, 15.0 , 15.5, 16.0 , 16.5, 17.0 , 17.5, 18.0 ,
                    18.5, 19.0 , 19.5, 20.0 , 20.5, 21.0 , 21.5, 22.0 , 22.5, 23.0 , 23.5,
                    24.0 , 24.5, 25.0 , 25.5
                ]
            },
            "blade_pitch": 0.0,
            "yaw_angle": 0.0,
            "tilt_angle": 0.0,
            "TSR": 8.0
        }
    }

    def __init__(self,
                 meta_dict=default_meta,
                 turbine_dict=default_turbine,
                 wake_dict=default_wake,
                 farm_dict=default_farm):
        self.base_meta = meta_dict
        self.base_turbine = turbine_dict
        self.base_wake = wake_dict
        self.base_farm = farm_dict

        self._meta_dict = self.build_meta_dict()
        self._turbine_dict = self.build_turbine_dict()
        self._wake_dict = self.build_wake_dict()
        self._farm_dict = self.build_farm_dict()

    def build_input_file_data(self):
        return {
            **self.meta_dict,
            "farm": self.farm_dict,
            "turbine": self.turbine_dict,
            "wake": self.wake_dict
        }

    def build_meta_dict(self):
        return {
            "type": self.input_or_default("type", self.base_meta, self.default_meta),
            "name": self.input_or_default("name", self.base_meta, self.default_meta),
            "description": self.input_or_default("description", self.base_meta, self.default_meta),
            "version": V1_0_0.version_string,
        }

    def build_farm_dict(self):
        name = self.input_or_default("name", self.base_farm, self.default_farm)
        description = self.input_or_default("description", self.base_farm, self.default_farm)

        _properties = self.input_or_default("properties", self.base_farm, self.default_farm)
        properties = {}
        properties["wind_speed"] = self.input_or_default("wind_speed", _properties, self.default_farm["properties"])
        properties["wind_direction"] = self.input_or_default("wind_direction", _properties, self.default_farm["properties"])
        properties["turbulence_intensity"] = self.input_or_default("turbulence_intensity", _properties, self.default_farm["properties"])
        properties["wind_shear"] = self.input_or_default("wind_shear", _properties, self.default_farm["properties"])
        properties["wind_veer"] = self.input_or_default("wind_veer", _properties, self.default_farm["properties"])
        properties["air_density"] = self.input_or_default("air_density", _properties, self.default_farm["properties"])
        properties["layout_x"] = self.input_or_default("layout_x", _properties, self.default_farm["properties"])
        properties["layout_y"] = self.input_or_default("layout_y", _properties, self.default_farm["properties"])

        return {
            "farm": {
                "type": "farm",
                "name": name,
                "description": description,
                "properties": properties
            }
        }

    def build_wake_dict(self):
        name = self.input_or_default("name", self.base_wake, self.default_wake)
        description = self.input_or_default("description", self.base_wake, self.default_wake)

        _properties = self.input_or_default("properties", self.base_wake, self.default_wake)
        _default_properties = self.default_wake["properties"]

        _parameters = self.input_or_default("parameters", _properties, self.default_wake["properties"])
        _default_parameters = self.default_wake["properties"]["parameters"]

        _turbulence_intensity = self.input_or_default("turbulence_intensity", _parameters, _default_parameters)
        _default_turbulence_intensity = _default_parameters["turbulence_intensity"]
        turbulence_intensity = {}
        turbulence_intensity["initial"] = self.input_or_default("initial", _turbulence_intensity, _default_turbulence_intensity)
        turbulence_intensity["constant"] = self.input_or_default("constant", _turbulence_intensity, _default_turbulence_intensity)
        turbulence_intensity["ai"] = self.input_or_default("ai", _turbulence_intensity, _default_turbulence_intensity)
        turbulence_intensity["downstream"] = self.input_or_default("downstream", _turbulence_intensity, _default_turbulence_intensity)

        _jensen = self.input_or_default("jensen", _parameters, _default_parameters)
        _default_jensen = _default_parameters["jensen"]
        jensen = {}
        jensen["we"] = self.input_or_default("we", _jensen, _default_jensen)

        _multizone = self.input_or_default("multizone", _parameters, _default_parameters)
        _default_multizone = _default_parameters["multizone"]
        multizone = {}
        multizone["me"] = self.input_or_default("me", _multizone, _default_multizone)
        multizone["we"] = self.input_or_default("we", _multizone, _default_multizone)
        multizone["aU"] = self.input_or_default("aU", _multizone, _default_multizone)
        multizone["bU"] = self.input_or_default("bU", _multizone, _default_multizone)
        multizone["mU"] = self.input_or_default("mU", _multizone, _default_multizone)

        _gauss = self.input_or_default("gauss", _parameters, _default_parameters)
        _default_gauss = _default_parameters["gauss"]
        gauss = {}
        gauss["ka"] = self.input_or_default("ka", _gauss, _default_gauss)
        gauss["kb"] = self.input_or_default("kb", _gauss, _default_gauss)
        gauss["alpha"] = self.input_or_default("alpha", _gauss, _default_gauss)
        gauss["beta"] = self.input_or_default("beta", _gauss, _default_gauss)
        gauss["ad"] = self.input_or_default("ad", _gauss, _default_gauss)
        gauss["bd"] = self.input_or_default("bd", _gauss, _default_gauss)

        _jimenez = self.input_or_default("jimenez", _parameters, _default_parameters)
        _default_jimenez = _default_parameters["jimenez"]
        jimenez = {}
        jimenez["kd"] = self.input_or_default("kd", _jimenez, _default_jimenez)
        jimenez["ad"] = self.input_or_default("ad", _jimenez, _default_jimenez)
        jimenez["bd"] = self.input_or_default("bd", _jimenez, _default_jimenez)

        _curl = self.input_or_default("curl", _parameters, _default_parameters)
        _default_curl = _default_parameters["curl"]
        curl = {}
        curl["model_grid_resolution"] = self.input_or_default("model_grid_resolution", _curl, _default_curl)
        curl["initial_deficit"] = self.input_or_default("initial_deficit", _curl, _default_curl)
        curl["dissipation"] = self.input_or_default("dissipation", _curl, _default_curl)
        curl["veer_linear"] = self.input_or_default("veer_linear", _curl, _default_curl)

        parameters = {
            "turbulence_intensity": turbulence_intensity,
            "jensen": jensen,
            "multizone": multizone,
            "gauss": gauss,
            "jimenez": jimenez,
            "curl": curl
        }

        properties = {
            "velocity_model": self.input_or_default("velocity_model", _properties, _default_properties),
            "deflection_model": self.input_or_default("deflection_model", _properties, _default_properties),
            "combination_model": self.input_or_default("combination_model", _properties, _default_properties),
            "parameters": parameters
        }
        
        return {
            "wake": {
                "type": "wake",
                "name": name,
                "description": description,
                "properties": properties
            }
        }

    def build_turbine_dict(self):

        name = self.input_or_default("name", self.base_turbine, self.default_turbine)
        description = self.input_or_default("description", self.base_turbine, self.default_turbine)

        _properties = self.base_turbine["properties"]
        properties = {}
        properties["rotor_diameter"] = self.input_or_default("rotor_diameter", _properties, self.default_turbine["properties"])
        properties["hub_height"] = self.input_or_default("hub_height", _properties, self.default_turbine["properties"])
        properties["blade_count"] = self.input_or_default("blade_count", _properties, self.default_turbine["properties"])
        properties["pP"] = self.input_or_default("pP", _properties, self.default_turbine["properties"])
        properties["pT"] = self.input_or_default("pT", _properties, self.default_turbine["properties"])
        properties["generator_efficiency"] = self.input_or_default("generator_efficiency", _properties, self.default_turbine["properties"])

        _power_thrust_table = _properties["power_thrust_table"]
        _default_power_thrust_table = self.default_turbine["properties"]["power_thrust_table"]
        power_thrust_table = {}
        power_thrust_table["power"] = self.input_or_default("power", _power_thrust_table, _default_power_thrust_table)
        power_thrust_table["thrust"] = self.input_or_default("thrust", _power_thrust_table, _default_power_thrust_table)
        power_thrust_table["wind_speed"] = self.input_or_default("wind_speed", _power_thrust_table, _default_power_thrust_table)
        properties["power_thrust_table"] = power_thrust_table

        properties["blade_pitch"] = self.input_or_default("blade_pitch", _properties, self.default_turbine["properties"])
        properties["yaw_angle"] = self.input_or_default("yaw_angle", _properties, self.default_turbine["properties"])
        properties["tilt_angle"] = self.input_or_default("tilt_angle", _properties, self.default_turbine["properties"])
        properties["TSR"] = self.input_or_default("TSR", _properties, self.default_turbine["properties"])

        return {
            "turbine": {
                "type": "turbine",
                "name": name,
                "description": description,
                "properties": properties
            }
        }
    
    @property
    def meta_dict(self):
        return self._meta_dict

    @property
    def turbine_dict(self):
        return self._turbine_dict

    @property
    def wake_dict(self):
        return self._wake_dict

    @property
    def farm_dict(self):
        return self._farm_dict
