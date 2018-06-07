"""
Script to process wind farm & turbine csv downloaded from: https://eerscmap.usgs.gov/uswtdb/data/
Metadata description can be found at https://eerscmap.usgs.gov/uswtdb/assets/data/uswtdb_v1_0_20180419.xml

Disregarding the following attributes for now:
t_rsa turbine rotor swept area square meters
t_ttlh turbine total height
t_conf_atr attribute confidence (0-N/A, 1=low, 2=partial, 3=full confidence)
t_conf_loc location confidence

ASSUMPTIONS / PROCESS
    * The script counts rows of spreadsheet to determine number of turbines instead of relying on p_tnum field
    * Don't generate input files when data are missing ('missing' for text fields, -9999 for numeric fields)

FOR USAGE:
python farm_processor.py -h

"""

import json, csv
import os, sys, getopt
import copy
import utm

# path to intermediate file to save processed CSV data
intermediate_file = 'farms.json'
# path to json input file template
template_file = '../inputFormApp/inputFiles/example_input_file.json'
generated_files_directory = './generatedInputFiles'

def main(argv):
    filename = None
    try:
        opts, args = getopt.getopt(argv,"hgi:a:",["all=","inputfile="])
    except getopt.GetoptError:
        print('Error')
        print('Process csv: farm_processor.py -i <inputfile.csv>')
        print('Generate input files from processed csv: farm_processor.py -g')
        print('Do both: farm_processor.py -i <inputfile.csv> -a')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('help')
            print('Process csv: farm_processor.py -i <inputfile.csv>')
            print('Generate input files from processed csv: farm_processor.py -g')
            print('Do all: farm_processor.py -a <inputfile.csv>')
            sys.exit()
        elif opt in ("-i", "--inputfile"):
            filename = arg
            print("PROCESS CSV: ", filename)
            process(filename)
        elif opt in ("-g"):
            print("GENERATE INPUT FILES FROM Intermediate file")
            generate()
        elif opt in ("-a", "--all"):
            filename = arg
            print("PROCESS CSV AND GENERATE INPUT FILES,  CSV: ", filename)
            process(filename)
            generate()


def process(filename):
   
    farms = [] 
    farm_cnt = 0
    turbine_cnt = 0
    current_farm = {'name': None, 'layout_x': [], 'layout_y': [], 'number_turbines': 0, 'turbine': {}, 'missing_data': False}

    # use p_name for farm name
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:

            # check if we are still processing the same farm
            if not(row['p_name'] == current_farm['name']):
                farm_cnt += 1
                # save old
                if current_farm['name'] is not None:
                    # make machine name, strip out spaces, (), and / 
                    current_farm['name'] = current_farm['name'].lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
                    farms.append(current_farm)

                # new farm
                turbine_cnt = 0 # reset turbine count
                current_farm = {}
                current_farm['missing_data'] = False
                current_farm['name'] = row['p_name']
                current_farm['number_turbines'] = 1
                # current_farm['number_turbines'] = row['p_tnum']
                # put p_cap in farm description (overall capacity in MW)
                current_farm['description'] = row['p_name'];
                if row['p_cap'] is not '-9999':
                    current_farm['description'] = current_farm['description'] + ' - ' + row['p_cap'] + ' MW'
                current_farm['description'] = current_farm['description']  + ' - ' + row['t_county'] + ', ' + row['t_state']
                # convert lat/long to x/y
                x,y, _, _ = utm.from_latlon(float(row['ylat']), float(row['xlong']))
                current_farm['layout_x'] = [x]
                current_farm['layout_y'] = [y]

                # turbine
                # use t_manu, t_model and t_cap (rated capacity in kW) to make turbine name & description
                # use xlong and ylat for layout array (convert to meters TBD)
                # store t_hh (hub height in m), t_rd (rotor diameter in m)
                current_farm['turbine'] = {}
                current_farm['turbine']['name'] = (row['t_manu'] + ' ' + row['t_model']).lower().replace(' ', '_')
                current_farm['turbine']['description'] = row['t_manu'] + ' ' + row['t_model'] + ' - ' + row['t_cap'] + ' kW'
                current_farm['turbine']['hub_height'] = float(row['t_hh'])
                current_farm['turbine']['rotor_diameter'] = float(row['t_rd'])

                # check for missing data (don't create input files for these)
                if row['t_manu'] == 'missing' or row['t_model'] == 'missing' or row['t_cap'] == '-9999':
                    # TODO: these are just descriptive data...may want to generate input file anyway
                    current_farm['missing_data'] = True
                if current_farm['turbine']['hub_height'] == -9999 or current_farm['turbine']['rotor_diameter'] == -9999 or float(row['xlong']) == -9999 or float(row['ylat']) == -9999:
                    # these data are used for non-descriptive fields in the input_file
                    current_farm['missing_data'] = True
            else:
                current_farm['number_turbines'] += 1
                x,y, _, _ = utm.from_latlon(float(row['ylat']), float(row['xlong']))
                current_farm['layout_x'].append(x)
                current_farm['layout_y'].append(y)    

    # save last farm            
    farms.append(current_farm)            

    # save array to json
    farms_json = {'farm_count': farm_cnt, 'farms': farms}
    with open(intermediate_file, 'w') as outfile:
        json.dump(farms_json, outfile, sort_keys=True, indent=2, separators=(',', ': '))

    print('DONE processing csv')

def generate():
    
    # load template
    with open(template_file, 'r') as f:
        template = json.load(f)

    # load intermediate file
    with open(intermediate_file, 'r') as f:
        data = json.load(f)

    # check if directory exists
    if not os.path.exists(generated_files_directory):
        os.makedirs(generated_files_directory)    

    # replace relevant values in template
    for farm in data['farms']:
        if farm['missing_data'] is False:    
            new_file = copy.deepcopy(template)
            new_file['description'] = farm['description'] + ' FLORIS Input File'
            # farm
            new_file['farm']['description'] = farm['description']
            new_file['farm']['name'] = farm['name']
            new_file['farm']['properties']['layout_x'] = farm['layout_x']
            new_file['farm']['properties']['layout_y'] = farm['layout_y']
            # turbine
            new_file['turbine']['description'] = farm['turbine']['description']
            new_file['turbine']['name'] = farm['turbine']['name']
            new_file['turbine']['properties']['hub_height'] = farm['turbine']['hub_height']
            new_file['turbine']['properties']['rotor_diameter'] = farm['turbine']['rotor_diameter']

            # save (use farm machine name as filename)
            with open('./generatedInputFiles/' + new_file['farm']['name'] + '.json', 'w') as f:
                json.dump(new_file, f,sort_keys=True, indent=2, separators=(',', ': '))

    print("DONE generating input files")

if __name__ == "__main__":
   main(sys.argv[1:])        