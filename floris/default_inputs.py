
default_inputs = {
    "name": "FLORIS defaults",
    "description": "Gauss-Curl hybrid model (GCH), 1 turbine, 8 m/s, 270 deg, 10% TI",
    "floris_version": "v4",

    "logging": {
        "console": {
            "enable": "true",
            "level": "WARNING"
        },
        "file": {
            "enable": "false",
            "level": "WARNING"
        },
    },

    "solver": {
        "type": "turbine_grid",
        "turbine_grid_points": 3
    },

    "farm": {
        "layout_x": [0.0],
        "layout_y": [0.0],
        "turbine_type": ["nrel_5MW"]
    },

    "flow_field": {
        "air_density": 1.225,
        "reference_wind_height": -1,
        "turbulence_intensities": [0.06],
        "wind_directions": [270.0],
        "wind_shear": 0.12,
        "wind_speeds": [8.0],
        "wind_veer": 0.0,
        "multidim_conditions": {
            "Tp": 2.5,
            "Hs": 3.01
        }
    },

    "wake": {
        "model_strings": {
            "velocity_model": "gauss",
            "deflection_model": "gauss",
            "combination_model": "sosfs",
            "turbulence_model": "crespo_hernandez",
        },

        "enable_secondary_steering": True,
        "enable_yaw_added_recovery": True,
        "enable_active_wake_mixing": False,
        "enable_transverse_velocities": True,

        "wake_deflection_parameters": {
            "gauss": {
                "ad": 0.0,
                "alpha": 0.58,
                "bd": 0.0,
                "beta": 0.077,
                "dm": 1.0,
                "ka": 0.38,
                "kb": 0.004
            },
            "jimenez": {
                "ad": 0.0,
                "bd": 0.0,
                "kd": 0.05,
            },
            "empirical_gauss": {
                "horizontal_deflection_gain_D": 3.0,
                "vertical_deflection_gain_D": -1,
                "deflection_rate": 22,
                "mixing_gain_deflection": 0.0,
                "yaw_added_mixing_gain": 0.0
            },
        },

        "wake_velocity_parameters": {
            "gauss": {
                "alpha": 0.58,
                "beta": 0.077,
                "ka": 0.38,
                "kb": 0.004
            },
            "jensen": {
                "we": 0.05,
            },
            "cc": {
                "a_s": 0.179367259,
                "b_s": 0.0118889215,
                "c_s1": 0.0563691592,
                "c_s2": 0.13290157,
                "a_f": 3.11,
                "b_f": -0.68,
                "c_f": 2.41,
                "alpha_mod": 1.0
            },
            "turbopark": {
                "A": 0.04,
                "sigma_max_rel": 4.0
            },
            "turboparkgauss": {
                "A": 0.04,
                "include_mirror_wake": True
            },
            "empirical_gauss": {
                "wake_expansion_rates": [0.023, 0.008],
                "breakpoints_D": [10],
                "sigma_0_D": 0.28,
                "smoothing_length_D": 2.0,
                "mixing_gain_velocity": 2.0,
                "awc_wake_exp": 1.2,
                "awc_wake_denominator": 400
            },
        },

        "wake_turbulence_parameters": {
            "crespo_hernandez": {
                "initial": 0.1,
                "constant": 0.5,
                "ai": 0.8,
                "downstream": -0.32
            },
            "wake_induced_mixing": {
                "atmospheric_ti_gain": 0.0
            }
        },
    }
}
