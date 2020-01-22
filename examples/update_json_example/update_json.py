import json
import glob
import copy

# This needs a proper home, but for now leaving here until we do the final pull to develop...

def update_json(filename, template_filename='template_input.json', output_filename=None):

    # If the output filename not given, just append out
    if output_filename is None:
        idx = filename.index('.')
        output_filename = filename[:idx] + '_out' + filename[idx:]
        print(output_filename)

    # Open the original json input
    with open(filename) as json_file:  
        data_old = json.load(json_file)

        # Open the template json input
        with open(template_filename) as json_file_template:  
            data = json.load(json_file_template)

            # Update the template values
            # Copy top matter, farm and turbine directly
            data['type'] = data_old['type']
            data['name'] = data_old['name']
            data['description'] = data_old['description']
            data['farm'] = data_old['farm']
            data['turbine'] = data_old['turbine']

            # Update wake parameters
            data['wake']['type'] = data_old['wake']['type']
            data['wake']['name'] = data_old['wake']['name']
            data['wake']['description'] = data_old['wake']['description']

            # Dig into properties, note turbulence goes to what's in template
            data['wake']['properties']['velocity_model'] = data_old['wake']['properties']['velocity_model']
            data['wake']['properties']['deflection_model'] = data_old['wake']['properties']['deflection_model']
            data['wake']['properties']['combination_model'] = data_old['wake']['properties']['combination_model']

            # Into parameters
            data['wake']['properties']['parameters']['wake_velocity_parameters']['jensen'] = data_old['wake']['properties']['parameters']['jensen']
            data['wake']['properties']['parameters']['wake_velocity_parameters']['multizone'] = data_old['wake']['properties']['parameters']['multizone']
            data['wake']['properties']['parameters']['wake_velocity_parameters']['gauss'] = copy.deepcopy(data_old['wake']['properties']['parameters']['gauss'])
            
            # Remove turbulence terms from gauss
            del data['wake']['properties']['parameters']['wake_velocity_parameters']['gauss']['ad']
            del data['wake']['properties']['parameters']['wake_velocity_parameters']['gauss']['bd']


            data['wake']['properties']['parameters']['wake_velocity_parameters']['jimenez'] = data_old['wake']['properties']['parameters']['jimenez']

            data['wake']['properties']['parameters']['wake_velocity_parameters']['curl'] = data_old['wake']['properties']['parameters']['curl']

            # Add turbulence terms to curl
            data['wake']['properties']['parameters']['wake_velocity_parameters']['curl']['initial'] = data_old['wake']['properties']['parameters']['turbulence_intensity']['initial']
            data['wake']['properties']['parameters']['wake_velocity_parameters']['curl']['constant'] = data_old['wake']['properties']['parameters']['turbulence_intensity']['constant']
            data['wake']['properties']['parameters']['wake_velocity_parameters']['curl']['ai'] = data_old['wake']['properties']['parameters']['turbulence_intensity']['ai']
            data['wake']['properties']['parameters']['wake_velocity_parameters']['curl']['downstream'] = data_old['wake']['properties']['parameters']['turbulence_intensity']['downstream']

            # Set the turbulence terms
            data['wake']['properties']['parameters']['wake_turbulence_parameters']['gauss'] = data_old['wake']['properties']['parameters']['turbulence_intensity']

            # Wake deflection parameters
            data['wake']['properties']['parameters']['wake_deflection_parameters']['gauss'] = data_old['wake']['properties']['parameters']['gauss']

            data['wake']['properties']['parameters']['wake_deflection_parameters']['jimenez'] = data_old['wake']['properties']['parameters']['jimenez']

            #Save the new file
            with open(output_filename, 'w') as outfile:  
                json.dump(data, outfile,indent='\t')


filename = 'example_input.json'
update_json(filename)