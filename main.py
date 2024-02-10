import argparse
import random
import numpy as np
import torch
import yaml
import os

import initialize_agents_networks
import model
from utils import get_time_stamped_outdir



# Parsing command line arguments
parser = argparse.ArgumentParser(description='PyTorch ABM Model for Covid 19.')
parser.add_argument('-p', '--params', help='Name of the yaml file with the parameters.')
parser.add_argument('-s', '--seed', type=int, help='Seed for python random, numpy and torch', default = 6666)
parser.add_argument('-n', '--num_runs', type=int, help='Number of runs', default = 1)
parser.add_argument('-y', '--num_steps', type=int, help='Number of steps', default = 180)
parser.add_argument('-a', '--app_adoption_rate', type=float, help='prob of active users of app in dct', default = 0.7)
parser.add_argument('-b', '--mct_reachable_prob', type=float, help='prob of agent reachable in mct', default = 0.8)
parser.add_argument('-d', '--dct_poc_comply_prob', type=float, help='prob of actually taking poc test in dct', default = 0.7)
parser.add_argument('-e', '--mct_poc_comply_prob', type=float, help='prob of actually taking poc test in mct', default = 0.8)
parser.add_argument('-r', '--mct_recall_prob', type=float, help='prob of recalling the contacts', default = 0.7)
parser.add_argument('-q', '--quarantine_break_prob', type=float, help='prob of breaking self-quarantine', default = 0.01)
parser.add_argument('-o', '--en_quarantine_enter_prob', type=float, help='prob of breaking self-quarantine', default = 0.8)
parser.add_argument('-v', '--vaccine_start_date', type=int, help='date of starting vaccination', default = 10)
parser.add_argument('-x', '--rtpcr_test_start_date', type=int, help='date of starting tetsing', default = 0)
parser.add_argument('-g', '--vaccine_first_dose_effectivness', type=float, help='first dose effectiveness', default = 0.9)
parser.add_argument('-z', '--vaccine_daily_production', type=int, help='daily production', default = 300)
parser.add_argument('-f', '--results_file_postfix', help='Postfix to be appended to output dir for ease of interpretation', default = '')
parser.add_argument('-l', '--log_full_dynamics', action='store_true', help='if logging infected agents for geography plots', default = False)
args = parser.parse_args()

# Setting seed
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
params['app_adoption_rate'] = args.app_adoption_rate
params['mct_reachable_prob'] = args.mct_reachable_prob
params['dct_poc_comply_prob'] = args.dct_poc_comply_prob
params['mct_poc_comply_prob'] = args.mct_poc_comply_prob
params['mct_recall_prob'] = args.mct_recall_prob
params['quarantine_break_prob'] = args.quarantine_break_prob
params['en_quarantine_enter_prob'] = args.en_quarantine_enter_prob
params['results_file_postfix'] = args.results_file_postfix
params['num_steps'] = args.num_steps
params['vaccine_start_date'] = args.vaccine_start_date
params['vaccine_first_dose_effectivness'] = args.vaccine_first_dose_effectivness
params['vaccine_daily_production'] = args.vaccine_daily_production
params['rtpcr_test_start_date'] = args.rtpcr_test_start_date
params['log_full_dynamics'] = args.log_full_dynamics


# if params['use_gpu']:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# else:
device = torch.device("cpu")
#Optionally re-initializing agents and re-building networks
print(params)
if params['type'] == "initialization":
    print("Calling initialization..")
    initialize_agents_networks.initialize_agents_networks(params)

    print("Initialization done.. ", " exiting")
    exit()
    print(" --- I SHOULD NOT BE HERE ----")

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
    # for i in range(5):
        abm.step()
    df = abm.collect_results()
    outfile = os.path.join(outdir, '{}_agents_stages_summary_seed_{}.csv'.format(n, params['seed']))
    df.to_csv(outfile, index=False)

    print(" ENDED RUN {}".format(n))
    print("-"*60)
