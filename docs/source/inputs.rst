Inputs
------

.. contents:: Contents
    :local:
    :backlinks: none

Configuration files for FLORIS are stored in the ``.json`` format, which is then parsed by :py:class:`floris.simulation.input_reader.InputReader` and stored in python dictionaries. The file begins with file type, name of the FLORIS configuration, and a brief description of the FLORIS configuration:

::

    "type": "floris_input",
    "name": "floris_input_file_Example",
    "description": "Example FLORIS Input file",

Farm
==========

The general description is followed by the ``farm`` dictionary. Here is where you can set the farm's atmospheric conditions, including:

    -   **wind_speed**: The incoming wind speed for the farm (m/s).
    -   **wind_direction**: The incoming wind direction for the farm (degrees) with cardinal settings (0 deg is North).
    -   **turbulence_intensity**: A decimal percent measure of turbulence intensity. Note: the models are not constructed for TIs above 0.14.
    -   **wind_shear**: The shear coefficient used in the log law to define the vertical velocity profile.
    -   **wind_veer**: A measure of the change in wind direction from the bottom of the rotor to the top of the rotor (degrees).
    -   **air_density**: The air density of the farm (kg/m^3).

You also set the initial farm layout and measurement locations for atmospheric
inputs in this dictionary:

    -   **layout_x**: The x-coordinates of the turbines (m).
    -   **layout_y**: The y-coordinates of the turbines (m).
    -   **wind_x**: The x-coordinates of the wind_speed, wind_direction, and turbulence_intensity measurements (m).
    -   **wind_y**: The y-coordinates of the wind_speed, wind_direction, and turbulence_intensity measurements (m).


::

  "farm": {
    "type": "farm",
    "name": "farm_example_2x2",
    "description": "Example 2x2 Wind Farm",
    "properties": {
      "wind_x":[0.0]
      "wind_y":[0.0]
      "wind_speed": [8.0],
      "wind_direction": [270.0],
      "turbulence_intensity": [0.06],
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
  },

Turbine
=======

After the ``farm`` dictionary is the ``turbine`` dictionary, which contains the relavant parameters for the turbine model being used. These are stored in the sub-dictionary ``properties``:

    -   **rotor_diameter**: Diameter of the turbine rotor (m).
    -   **hub_height**: Hub height of the turbine (m).
    -   **blade_count**: The number of blades on the turbine.
    -   **pP**: Exponent used on the cosine power loss due to yaw.
    -   **pT**: Exponent used on the cosine power loss due to tilt.
    -   **generator_efficiency**: The generator efficiency used in calculating
        power. If the Cp (``power``) data is electrical and not aerodynamic,
        then set ``generator_efficiency = 1.0``.
    -   **power_thrust_table**: Sub-dictionary containing Cp, Ct, and wind
        speed data.

        -   **power**: A list of Cp values that correspond to the Ct
            (``thrust``) and ``wind_speed`` data.
        -   **thrust**: A list of Ct values that correspond to the Cp
            (``power``) and ``wind_speed`` data.
        -   **wind_speed**: A list of wind speeds that correspond to the Cp
            (``power``) and Ct (``thrust``) data.

    -   **blade_pitch**: The pitch of the turbine blades to run the simulation at. Not currently implemented; planned for future use.
    -   **yaw_angle**: The initial yaw angle for all the turbines (degrees).
    -   **tilt_angle**: The tilt angle of the rotor (degrees).
    -   **TSR**: The tip-speed ratio of the turbine.

::

    "turbine": {
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
            "power": [
            0.0,
            0.15643578,
            ...,
            0.04889237,
            0.0
            ],
            "thrust": [
            1.10610965,
            1.09515807,
            ...,
            0.05977061,
            0.0
            ],
            "wind_speed": [
            0.0,
            2.5,
            ...,
            24.56926707,
            30.0
            ]
        },
        "blade_pitch": 0.0,
        "yaw_angle": 0.0,
        "tilt_angle": 0.0,
        "TSR": 8.0
        }
    },

Wake
====
The ``wake`` dictionary contains parameter values for the different wake models available in FLORIS. These parameter values are stored in the ``properties`` sub-dictionary:

    -   **velocity_model**: The name of the velocity model to use. Possible
        models include "jensen", "multiZone", "gauss", and "curl".
    -   **deflection_model**: The name of the deflection model to use. Possible
        models include "jimenez", "gauss", and "curl".
    -   **combination_model**: The method to use to combine the base flow field
        and the turbine wakes. Possible options include "fls" (freestream
        linear superposition) and "sosfs" (sum of squares freestream
        superposition).
    -   **parameters**: Sub-dictionary containing parameters for the different
        wake models.

        -   **turbulence_intensity**: Sub-dictionary containing turbulence
            intensity parameters for turbulence added by turbines.

            -   **initial**: Initial turbulence intensity.
            -   **constant**: Coefficient for the turbulence model.
            -   **ai**: Exponent for the turbine's axial induction factor.
            -   **downstream**: Exponent for the ratio of the distance
                downstream to the turbine rotor's diameter.

        -   **jensen**: Sub-dictionary for the Jensen wake model.

            -   **we**: A float that is the linear wake decay constant that
                defines the cone boundary for the wake as well as the velocity
                deficit.

        -   **multizone**: Sub-dictionary for the MultiZone model.

            -   **me**: A list of three floats that help determine the slope of
                the diameters of the three wake zones (near wake, far wake,
                mixing zone) as a function of downstream distance.
            -   **we**: A float that is the scaling parameter used to adjust
                the wake expansion, helping to determine the slope of the
                diameters of the three wake zones as a function of downstream
                distance, as well as the recovery of the velocity deficits in
                the wake as a function of downstream distance.
            -   **aU**: A float that is a parameter used to determine the
                dependence of the wake velocity deficit decay rate on the rotor
                yaw angle.
            -   **bU**: A float that is another parameter used to determine the
                dependence of the wake velocity deficit decay rate on the rotor
                yaw angle.
            -   **mU**: A list of three floats that are parameters used to
                determine the dependence of the wake velocity deficit decay

        -   **gauss**: Sub-dictionary for the Gaussian model.

            -   **ka**: A float that is a parameter used to determine the
                linear relationship between the turbulence intensity and the
                width of the Gaussian wake shape.
            -   **kb**: A float that is a second parameter used to determine
                the linear relationship between the turbulence intensity and
                the width of the Gaussian wake shape.
            -   **alpha**: A float that is a parameter that determines the
                dependence of the downstream boundary between the near wake and
                far wake region on the turbulence intensity.
            -   **beta**: A float that is a parameter that determines the
                dependence of the downstream boundary between the near wake and
                far wake region on the turbine's induction factor.
            -   **ad**: Parameter to include additional deflection of the wake
                to account for delfection due to the rotation of the turbine.
            -   **bd**: Parameter to include additional deflection of the wake
                to account for delfection due to the rotation of the turbine.

        -   **jimenez**: Sub-dictionary for the Jimenez wake deflection model.

            -   **kd**:
            -   **ad**:
            -   **bd**:

        -   **curl**: Sub-dictionary for the Curled Wake model.

            -   **model_grid_resolution**: A list of three values that
                represent the number of divisions of the flow field over the
                x-, y-, and z-domains.
            -   **initial_deficit**: A float that scales the initial wake
                deficit profile spawned at each turbine.
            -   **dissipation**: A tuning parameter that scales the effective
                viscosity of the flow, essentially changing how quickly the
                wakes dissipate.
            -   **veer_linear**: The amount of veer for the curled wake model.

::

    "wake": {
        "type": "wake",
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
    }

.. _sample_input_file_ref:

Sample Input File
=================

Below is a sample FLORIS input file setup for a 2x2 wind turbine array modeling the NREL 5-MW wind turbine.
This file is also available on Github: `example_input.json <https://github.com/NREL/floris/blob/develop/examples/example_input.json>`_

.. literalinclude:: ../../examples/example_input.json
