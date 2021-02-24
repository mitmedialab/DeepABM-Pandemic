import argparse
import random
import numpy as np
import torch
import yaml
import os

import initialize_agents_networks
import model
from utils import get_time_stamped_outdir


def runner(flags_dict):
    '''Update using flags_dict'''

    # Parsing command line arguments
    parser = argparse.ArgumentParser(description='PyTorch ABM Model for Covid 19.')
    parser.add_argument('-p', '--params', help='Name of the yaml file with the parameters.', default='Data/generated_params-expvac-1.yaml')
    parser.add_argument('-s', '--seed', type=int, help='Seed for python random, numpy and torch', default=6666)
    parser.add_argument('-n', '--num_runs', type=int, help='Number of runs', default=1)
    parser.add_argument('-f', '--results_file_postfix',
                        help='Postfix to be appended to output dir for ease of interpretation', default='')
    args = parser.parse_args()

    #Setting seed
    print('Seed used for python random, numpy and torch is {}'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #Reading params
    with open(args.params, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Error in reading parameters file')
            print(exc)

    params['seed'] = args.seed
    params['num_runs'] = args.num_runs
    params['results_file_postfix'] = args.results_file_postfix

    # updating params based on flag dict
    params['use_quarantine_logic'] = flags_dict['Quarantine']
    params['use_vaccination_logic'] = flags_dict['Vaccination']
    params['use_den_logic'] = flags_dict['DEN']
    params['use_poc_test_logic'] = flags_dict['POC_Test']

    map_val = {True: 'Yes', False: 'No'}
    params['results_file_postfix'] = params['results_file_postfix'] + ''.join([map_val[flags_dict[ix]] + ix[:3] + '_' for ix in flags_dict.keys()])

    print("results_file_postfix is: ", params['results_file_postfix'])
    print("Updated parameters: ", params['use_quarantine_logic'], params['use_vaccination_logic'], params['use_den_logic'], params['use_poc_test_logic'])
    print("Returning now..")

    if params['use_gpu']:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    #Optionally re-initializing agents and re-building networks
    if params['type'] == "initialization":
        initialize_agents_networks.initialize_agents_networks(params)

    if params['use_quarantine_logic']:
        print('This is a case with quarantine logic')
    else:
        print('This is a case without quarantine logic')

    if params['use_den_logic']:
        print('This is a case with den logic')
    else:
        print('This is a case without den logic')

    outdir = get_time_stamped_outdir(
                    [params['output_location']['parent_dir'],
                        params['output_location']['results_dir']], params['results_file_postfix']
            )
    outfile = os.path.join(
            outdir,
            params['genrerated_params_file_name']
        )
    with open(outfile, 'w') as stream:
        try:
            yaml.dump(params, stream)
        except yaml.YAMLError as exc:
            print(exc)
    for n in range(params['num_runs']):
        abm = model.TorchABMCovid(params, device)
        for i in range(params['num_steps']):
            abm.step()
        df = abm.collect_results()
        outfile = os.path.join(outdir, '{}_agents_stages_summary_seed_{}.csv'.format(n, params['seed']))
        df.to_csv(outfile, index=False)

        print(" ENDED RUN {}".format(n))
        print("-"*60)

    return outdir, params
