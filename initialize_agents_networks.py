import random
import os
import numpy as np
from collections import Counter
import networkx as nx
from utils import custom_watts_strogatz_graph, get_dir_from_path_list
from itertools import combinations
import pandas as pd
from scipy.stats import nbinom
import yaml


def assign_age_ix_to_agents(age_ix_prob_list, num_agents):
    res = np.random.choice(len(age_ix_prob_list), p=age_ix_prob_list, size=num_agents)
    return res


def assign_stage_ix_to_agents(stage_ix_pop_dict):
    init_stages = []
    for st in stage_ix_pop_dict.keys():
        init_stages.extend([st]*stage_ix_pop_dict[st])
    random.shuffle(init_stages)
    return init_stages


def assign_household_ix_to_agents(households_sizes_list, households_sizes_prob_list, num_agents):
    household_id = 0
    total_agents_unassigned = num_agents
    agent_households = []
    household_agents = []
    last_agent_id = 0
    while total_agents_unassigned > 0:
        household_size = np.random.choice(households_sizes_list, 
                            p=households_sizes_prob_list)
        if (household_size > total_agents_unassigned):
            household_size = total_agents_unassigned
        agent_households.extend([household_id]*household_size)
        household_id += 1
        total_agents_unassigned -= household_size
        household_agents.append(list(range(last_agent_id, last_agent_id+household_size)))
        last_agent_id += household_size
    return agent_households, household_agents


def assign_occupation_ix_to_agents(agents_ages, occupations_sizes_prob_list, elderly_ix, child_ix, child_upper_ix, adult_upper_ix): #from enum
    agents_occupations = []
    for age in agents_ages:
        if age <= child_upper_ix: 
            agents_occupations.append(child_ix)
        elif age <= adult_upper_ix:
            agents_occupations.append(np.random.choice(len(occupations_sizes_prob_list), p=occupations_sizes_prob_list))
        else:
            agents_occupations.append(elderly_ix)
    return agents_occupations


def assign_app_user_status_to_agents(num_agents, app_adoption_rate, agents_ages, app_user_agewise_probs):
    total_app_users = app_adoption_rate*num_agents
    app_assigned = 0
    all_agents = list(range(num_agents))
    app_users = [False]*len(all_agents)
    while app_assigned < total_app_users:
        a_idx = random.choice(all_agents)
        age_group = agents_ages[a_idx]
        agent_app_prob = app_user_agewise_probs[age_group]
        get_app = np.random.choice([True, False], p = [agent_app_prob, 1 - agent_app_prob])
        if get_app:
            app_users[a_idx] = True
            all_agents.remove(a_idx)
            app_assigned += 1

    return app_users


def get_num_random_interactions(age, random_network_params_dict, child_upper_ix, adult_upper_ix):
    if age <= child_upper_ix:
        mean = random_network_params_dict['CHILD']['mu']
        sd = random_network_params_dict['CHILD']['sigma']
    elif age <= adult_upper_ix:
        mean = random_network_params_dict['ADULT']['mu']
        sd = random_network_params_dict['ADULT']['sigma']
    else:
        mean = random_network_params_dict['ELDERLY']['mu']
        sd = random_network_params_dict['ELDERLY']['sigma']
    p = mean / (sd*sd)
    n = mean * mean / (sd * sd - mean)
    num_interactions = nbinom.rvs(n, p)
    return num_interactions


def create_and_write_household_network(household_agents, path):
    household_network = nx.Graph()
    #Adding edges for all agents in each household
    for household in household_agents:
        h_edges = list(combinations(household, 2))
        household_network.add_edges_from(h_edges)
    outfile = os.path.join(get_dir_from_path_list(path), 'household.csv')
    nx.write_edgelist(household_network, outfile , delimiter = ',', data = False)


def create_and_write_occupation_networks(agents_occupations, occupations_names_list, 
occupations_ix_list, num_steps, occupation_nw_infile, path):
    if not os.path.isfile(occupation_nw_infile):
        print('The file with random network parameters not found at location {}'.format(occupation_nw_infile))
        raise FileNotFoundError
    occupation_nw_df = pd.read_csv(occupation_nw_infile, index_col=0)
    occupation_nw_parameters_dict = {
        a: {'mu': occupation_nw_df.loc[a, 'mu'], 'rewire': occupation_nw_df.loc[a, 'rewire']}
        for a in occupation_nw_df.index.to_list()
    }
    occupations_population = Counter(agents_occupations)
    occupations_agents = [[a for a in range(len(agents_occupations)) if agents_occupations[a] == o] for o in occupations_ix_list]
    for t in range(num_steps):
        occupation_networks = {}
        for occ in occupations_ix_list:
            n_interactions = occupation_nw_parameters_dict[occupations_names_list[occ]]['mu']
            network_rewire = occupation_nw_parameters_dict[occupations_names_list[occ]]['rewire']
            occupation_networks[occ] = custom_watts_strogatz_graph(
                occupations_population[occ],
                min(np.round(n_interactions, 0).astype(int), occupations_population[occ] - 1),
                [network_rewire, occupations_agents[occ]])
                
        for key in occupation_networks.keys():
            outfile = os.path.join(get_dir_from_path_list(path + [occupations_names_list[key]]), '{}.csv'.format(t))
            G = occupation_networks[key]
            nx.write_edgelist(G, outfile, delimiter = ',', data = False)


def create_and_write_random_networks(num_agents, agents_ages, num_steps, random_nw_infile, child_upper_ix, adult_upper_ix, path):
    if not os.path.isfile(random_nw_infile):
        print('The file with random network parameters not found at location {}'.format(random_infile))
        raise FileNotFoundError
    random_nw_df = pd.read_csv(random_nw_infile, index_col=0)
    random_network_params_dict = {
        a: {'mu': random_nw_df.loc[a, 'mu'], 'sigma': random_nw_df.loc[a, 'mu']}
        for a in random_nw_df.index.to_list()
    }
    agents_random_interactions = [get_num_random_interactions(age, random_network_params_dict, child_upper_ix, adult_upper_ix) for age in agents_ages]
    for t in range(num_steps):
        interactions_list = []
        for agent_id in range(num_agents):
            interactions_list.extend([agent_id]*agents_random_interactions[agent_id])
        random.shuffle(interactions_list)
        edges_list = [(interactions_list[i], interactions_list[i+1]) for i in range(len(interactions_list)-1)]
        G = nx.Graph()
        G.add_edges_from(edges_list)
        outfile = os.path.join(get_dir_from_path_list(path), '{}.csv'.format(t))
        nx.write_edgelist(G, outfile, delimiter = ',', data = False)


########################################################################################################
def initialize_agents_networks(params):
    '''
    There are 5 input files corresponding to age, stage, occupation, household size distribution and app user parameters.
    There are 2 more input files for occupation and random network construction parameters.
    Thus, there are 7 input files in all to initialize agents and build networks
    In addition to creating files in GeneratedAgentsData and GeneratedNetworks, this function 
    also creates a generated_params.yaml file which will be used by main and model
    '''
    #Reading agent data files and updating params
    infile = params['initialization_input_files']['agents_ages_filename']
    if not os.path.isfile(infile):
        print('The distribution of ages across population file not found at location {}'.format(infile))
        raise FileNotFoundError
    ages_df = pd.read_csv(infile, index_col=0)
    params['age_groups'] = ages_df.index.to_list()
    params['age_groups_to_ix_dict'] = {
        params['age_groups'][i] : i
        for i in range(len(params['age_groups']))
    }
    params['age_ix_to_group_dict'] = {
        params['age_groups_to_ix_dict'][k]: k
        for k in params['age_groups_to_ix_dict'].keys()
    }
    total_population_in_age_dist_params = ages_df['Number'].sum()
    params['age_ix_pop_dict'] = {
        k: int(ages_df.loc[params['age_ix_to_group_dict'][k]].values[0])
        for k in range(len(params['age_groups']))
    }
    params['age_ix_prob_list'] = [
        float(params['age_ix_pop_dict'][k]/total_population_in_age_dist_params)
        for k in range(len(params['age_groups']))
    ]
    params['CHILD_Upper_Index'] = 1 #TODO
    params['ADULT_Upper_Index'] = 6 #TODO

    infile = params['initialization_input_files']['agents_initial_stages_filename']
    if not os.path.isfile(infile):
        print('The file with initial stages of agents not found at location {}'.format(infile))
        raise FileNotFoundError
    stages_df = pd.read_csv(infile, index_col=0)
    params['stages'] = stages_df.index.to_list()
    params['num_stages'] = len(params['stages'])
    params['stages_to_ix_dict'] = {
        params['stages'][i]: i
        for i in range(len(params['stages']))
    }
    params['stage_ix_to_stages_dict'] = {
        params['stages_to_ix_dict'][k]: k 
        for k in params['stages_to_ix_dict'].keys()
    }
    params['num_agents'] = int(stages_df['Number'].sum())
    params['stage_ix_pop_dict'] = {
        k: int(stages_df.loc[params['stage_ix_to_stages_dict'][k]].values[0])
        for k in range(len(params['stages']))
    }

    infile = params['initialization_input_files']['agents_household_sizes_filename']
    if not os.path.isfile(infile):
        print('The file with household sizes distribution not found at location {}'.format(infile))
        raise FileNotFoundError
    households_df = pd.read_csv(infile, index_col=0)
    total_population_in_household_dist_params = households_df['Number'].sum()
    params['households_sizes_list'] = households_df.index.to_list()
    params['households_sizes_prob_list'] = [
        float(households_df.loc[k].values[0]/total_population_in_household_dist_params)
        for k in params['households_sizes_list']
    ]

    infile = params['initialization_input_files']['agents_occupations_filename']
    if not os.path.isfile(infile):
        print('The file with occupations distribution not found at location {}'.format(infile))
        raise FileNotFoundError
    occupations_df = pd.read_csv(infile, index_col=0)
    total_population_in_occupation_dist_params = occupations_df['Number'].sum()
    params['occupations_names_list'] = occupations_df.index.to_list()
    params['occupations_sizes_prob_list'] = [
        float(occupations_df.loc[k].values[0]/total_population_in_occupation_dist_params)
        for k in params['occupations_names_list']
    ]
    params['occupations_to_ix_dict'] = {
        i: params['occupations_names_list'][i] 
        for i in range(len(params['occupations_names_list']))
    }
    elderly_ix = len(params['occupations_names_list'])
    child_ix = len(params['occupations_names_list']) + 1
    params['occupations_names_list'].extend(['ELDERLY', 'CHILD'])
    params['occupations_to_ix_dict'][elderly_ix] = 'ELDERLY'
    params['occupations_to_ix_dict'][child_ix] = 'CHILD'
    params['occupation_ix_to_occupations_dict'] = {
        params['occupations_to_ix_dict'][k]: k 
        for k in params['occupations_to_ix_dict'].keys()
    }
    params['occupations_ix_list'] = list(range(len(params['occupations_names_list'])))

    #REading app-users parameter file
    infile = params['initialization_input_files']['agents_app_user_parameters_filename']
    if not os.path.isfile(infile):
        print('The file with app user parameters not found at location {}'.format(infile))
        raise FileNotFoundError
    appuserparams_df = pd.read_csv(infile, index_col=0)
    params['app_user_agewise_probs_dict'] = {
        i: float(appuserparams_df.loc[i].values[0])
        for i in appuserparams_df.index.to_list()
    }
    app_user_agewise_probs = [params['app_user_agewise_probs_dict'][
        params['age_ix_to_group_dict'][a]
    ]
        for a in range(len(params['age_groups']))
    ]
    
    #Creating agent dataframe
    agents_households, household_agents = assign_household_ix_to_agents(params['households_sizes_list'], 
                            params['households_sizes_prob_list'], params['num_agents'])
    agent_df = pd.DataFrame()
    agent_df['age_group'] = assign_age_ix_to_agents(params['age_ix_prob_list'], params['num_agents'])
    agent_df['stage'] = assign_stage_ix_to_agents(params['stage_ix_pop_dict'])
    agent_df['household'] = agents_households
    agent_df['occupation_network'] = assign_occupation_ix_to_agents(agent_df['age_group'].values.tolist(),
        params['occupations_sizes_prob_list'], elderly_ix, child_ix, params['CHILD_Upper_Index'], params['ADULT_Upper_Index'])
    agent_df['app_user'] = assign_app_user_status_to_agents(params['num_agents'], 
    params['app_adoption_rate'], agent_df['age_group'].values.tolist(), app_user_agewise_probs)
    outfile = os.path.join(get_dir_from_path_list([params['output_location']['parent_dir'], 
    params['output_location']['agents_dir']]), params['output_location']['agents_outfile'])
    agent_df.to_csv(outfile)

    #Creating household dataframe
    household_df = pd.DataFrame()
    household_df['Members'] = household_agents
    outfile = os.path.join(get_dir_from_path_list([params['output_location']['parent_dir'], 
    params['output_location']['agents_dir']]), params['output_location']['households_outfile'])
    household_df.to_csv(outfile)

    #Constructing networks (assuming agents are ids. Note that ids range from 0 to num_agents-1)
    create_and_write_household_network(household_agents, [params['output_location']['parent_dir'], 
                    params['output_location']['networks_dir'],
                    params['output_location']['household_networks_dir']])
    create_and_write_occupation_networks(agent_df['occupation_network'].values.tolist(), 
            params['occupations_names_list'], params['occupations_ix_list'], 
            params['num_steps'], 
            params['initialization_input_files']['occupation_nw_parameters_filename'], 
            [params['output_location']['parent_dir'], params['output_location']['networks_dir'],
                    params['output_location']['occupation_networks_dir']])
    create_and_write_random_networks(params['num_agents'], 
    agent_df['age_group'].values.tolist(), params['num_steps'], 
    params['initialization_input_files']['random_nw_parameters_filename'], 
    params['CHILD_Upper_Index'], params['ADULT_Upper_Index'], 
    [params['output_location']['parent_dir'], params['output_location']['networks_dir'],
                    params['output_location']['random_networks_dir']])


    #Write updated params file
    params['type'] = 'generated'
    outfile = os.path.join(
        get_dir_from_path_list([params['output_location']['parent_dir']]), 
        params['genrerated_params_file_name']
    )
    with open(outfile, 'w') as stream:
        try:
            yaml.dump(params, stream)
        except yaml.YAMLError as exc:
            print(exc)

