'use strict';

/* Controllers */

function InputsController($scope, $http, $log) {

    $scope.velocityModels = ['floris', 'gauss', 'jensen'];
    $scope.deflectionModels = ['gauss_deflection', 'jimenez'];
    $scope.wakeCombinations = ['sosfs'];

	$scope.inputs = {name: 'floris_input_file_Example', type: "floris input", description: 'Example FLORIS Input file'};
    $scope.inputs.farm = {
        type: "farm",
		name: "farm_example_2x2",
        description: "Example Wind Farm",
        properties: {
            wind_speed: 8.0,
            wind_direction: 270.0,
            turbulence_intensity: 0.1,
            wind_shear: 0.12,
            wind_veer: 0.0,
            air_density: 1.225,
            wake_combination: "sosfs",
            layout_x: [],
            layout_y: []
        }
    };
    $scope.inputs.turbine = {
		type: "turbine",
        name: "nrel_5mw",
		description: "NREL 5MW",
		properties: {
            rotor_diameter: 126.0,
			hub_height: 90.0,
			blade_count: 3,
			pP: 1.88,
			pT: 1.88,
			generator_efficiency: 1.0,
			eta: 0.768,
			power_thrust_table: {
            	power: [],
				thrust: [],
				wind_speed: []
            },
            blade_pitch: 1.9,
			yaw_angle: 20.0,
			tilt_angle: 0.0,
			TSR: 8.0
        }
    };

    $scope.inputs.wake = {
        type: "wake",
        name: "wake_default",
        description: "wake",
        properties: {
            velocity_model: "gauss",
            deflection_model: "jimenez",
            parameters: {
                jensen: {
                    we: 0.05
                },
                floris: {
                    me: [
                        -0.05,
                        0.3,
                        1.0
                    ],
                    aU: 12.0,
                    bU: 1.3,
                    mU: [
                        0.5,
                        1.0,
                        5.5
                    ]
                },
                gauss: {
                    ka: 0.38371,
                    kb: 0.004,
                    alpha: 0.58,
                    beta: 0.077
                },
                jimenez: {
                    kd: 0.17,
                    ad: -4.5,
                    bd: -0.01
                },
                gauss_deflection: {
                    alpha: 0.58,
                    beta: 0.077,
                    ad: 0.0,
                    bd: 0.0
                }
            }
        }
	};

    $log.log("INPUTS: ", $scope.inputs);

    $scope.saveFile = function() {

        $log.log("CALCULATE!");
        // TODO: call a formatting function here to get the file looking like you want
        const config = {headers: {Accept: 'application/json'}};
        $http.post('/generate/', $scope.inputs, config)
            .then(function(response){
                console.log("RESPONSE: ", response);
                // show success modal
                $('#successModal').modal({
                    keyboard: false
                })

            }, function(error){
                // TODO: SHOW ERROR ON SCREEN
                console.log(error);
        });
    };

    $scope.addTurbine = function() {
        $scope.inputs.farm.properties.layout_x.push(0);
        $scope.inputs.farm.properties.layout_y.push(0);
    }

    $scope.addEntry = function() {
        $scope.inputs.turbine.properties.power_thrust_table.power.push(0);
        $scope.inputs.turbine.properties.power_thrust_table.thrust.push(0);
        $scope.inputs.turbine.properties.power_thrust_table.wind_speed.push(0);
    }
}

