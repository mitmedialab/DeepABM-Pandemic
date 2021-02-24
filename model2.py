import os
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from utils import get_dir_from_path_list, make_one_hot, time_dists_split
from scipy.stats import gamma
from collections import deque
from torch_sparse import SparseTensor
from scipy.stats import nbinom

def get_num_random_interactions(age, mus, sigmas, child_upper_ix, adult_upper_ix):
    if age <= child_upper_ix:
        mean = mus[0]
        sd = sigmas[0]
    elif age <= adult_upper_ix:
        mean = mus[1]
        sd = mus[1]
    else:
        mean = mus[2]
        sd = mus[2]
    p = mean / (sd*sd)
    n = mean * mean / (sd * sd - mean)
    num_interactions = nbinom.rvs(n, p)
    return num_interactions

def lam(x_i, x_j, edge_attr, t, R, SFSusceptibility, SFInfector, lam_gamma_integrals):
        S_A_s = SFSusceptibility[x_i[:,0].long()]
        A_s_i = SFInfector[x_j[:,1].long()]
        B_n = edge_attr[1, :]
        integrals = torch.zeros_like(B_n)
        infected_idx = x_j[:, 2].bool()
        infected_times = t - x_j[infected_idx, 3]
        integrals[infected_idx] =  lam_gamma_integrals[infected_times.long()]   #:,2 is infected index and :,3 is infected time
        edge_network_numbers = edge_attr[0, :] #to account for the fact that mean interactions start at 4th position of x
        I_bar = torch.gather(x_i[:, 4:27], 1, edge_network_numbers.view(-1,1).long()).view(-1) #to account for the fact that mean interactions start at 4th position of x
        not_quarantined = torch.logical_not(x_j[:,28])
        # ego_agents_vaccine_protection = torch.distributions.Categorical(
        #     probs = torch.vstack((1-x_i[:,29], x_i[:,29])).t()
        # ).sample().view(-1).long()
        print(S_A_s.shape, A_s_i.shape, edge_attr[1].shape, integrals.shape, I_bar.shape) #, ego_agents_vaccine_protection.shape)
        # (R*S_A_s*A_s_i*B_n)/(\bar I) * integral
        # res = torch.logical_not(ego_agents_vaccine_protection)*not_quarantined*R*S_A_s*A_s_i*B_n*integrals/I_bar #Edge attribute 1 is B_n
        res = not_quarantined*R*S_A_s*A_s_i*B_n*integrals/I_bar #Edge attribute 1 is B_n
        return res.view(-1, 1)

class InfectionNetwork(MessagePassing):
    def __init__(self, lam, R, SFSusceptibility, SFInfector, lam_gamma_integrals):
        super(InfectionNetwork, self).__init__(aggr='add')
        self.lam = lam
        self.R = R
        self.SFSusceptibility = SFSusceptibility
        self.SFInfector = SFInfector
        self.lam_gamma_integrals = lam_gamma_integrals

    def forward(self, data):
        x = data.x 
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        t = data.t
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, t=t, 
                R=self.R, SFSusceptibility=self.SFSusceptibility, 
                SFInfector=self.SFInfector, lam_gamma_integrals=self.lam_gamma_integrals)

    def message(self, x_i, x_j, edge_attr, t, 
                        R, SFSusceptibility, SFInfector, lam_gamma_integrals):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        tmp = self.lam(x_i, x_j, edge_attr, t, 
            R, SFSusceptibility, SFInfector, lam_gamma_integrals)  # tmp has shape [E, 2 * in_channels]
        return tmp

class TorchABMCovid():
    def __init__(self, params, device):
        self.params = params
        self.device = device
        #**********************************************************************************
        #Environment state variables
        #**********************************************************************************
        #**********************************************************************************
        #Static
        self.agents_ix = torch.arange(0, self.params['num_agents']).long().to(self.device)
        infile = os.path.join(
                self.params['output_location']['parent_dir'],
                self.params['output_location']['agents_dir'],
                self.params['output_location']['agents_outfile'])
        agents_df = pd.read_csv(infile)
        self.agents_ages = torch.tensor(agents_df['age_group'].to_numpy()).long().to(self.device) #Assuming age remains constant in the simulation
        self.agents_households = torch.tensor(agents_df['household'].to_numpy()).long().to(self.device) #Assuming occupation remains constant in the simulation
        self.num_networks = 23
        self.network_type_dict = {}
        self.network_type_dict['household'] = 0
        self.network_type_dict['AGRICULTURE'] = 1
        self.network_type_dict['MINING'] = 2
        self.network_type_dict['UTILITIES'] = 3 
        self.network_type_dict['CONSTRUCTION'] = 4 
        self.network_type_dict['MANUFACTURING'] = 5 
        self.network_type_dict['WHOLESALETRADE'] = 6 
        self.network_type_dict['RETAILTRADE'] = 7 
        self.network_type_dict['TRANSPORTATION'] = 8 
        self.network_type_dict['INFORMATION'] = 9 
        self.network_type_dict['FINANCEINSURANCE'] = 10 
        self.network_type_dict['REALESTATERENTAL'] = 11 
        self.network_type_dict['SCIENTIFICTECHNICAL'] = 12 
        self.network_type_dict['ENTERPRISEMANAGEMENT'] = 13 
        self.network_type_dict['WASTEMANAGEMENT'] = 14 
        self.network_type_dict['EDUCATION'] = 15
        self.network_type_dict['HEALTHCARE'] = 16 
        self.network_type_dict['ART'] = 17
        self.network_type_dict['FOOD'] = 18 
        self.network_type_dict['OTHER'] = 19
        self.network_type_dict['CHILD'] = 20
        self.network_type_dict['ELDERLY'] = 21
        self.network_type_dict['random'] = 22

        self.network_type_dict_inv = {}
        self.network_type_dict_inv[0] = 'household'
        self.network_type_dict_inv[1] = 'AGRICULTURE'
        self.network_type_dict_inv[2] = 'MINING'
        self.network_type_dict_inv[3] = 'UTILITIES'
        self.network_type_dict_inv[4] = 'CONSTRUCTION'
        self.network_type_dict_inv[5] = 'MANUFACTURING'
        self.network_type_dict_inv[6] = 'WHOLESALETRADE'
        self.network_type_dict_inv[7] = 'RETAILTRADE'
        self.network_type_dict_inv[8] = 'TRANSPORTATION'
        self.network_type_dict_inv[9] = 'INFORMATION'
        self.network_type_dict_inv[10] = 'FINANCEINSURANCE'
        self.network_type_dict_inv[11] = 'REALESTATERENTAL'
        self.network_type_dict_inv[12] = 'SCIENTIFICTECHNICAL'
        self.network_type_dict_inv[13] = 'ENTERPRISEMANAGEMENT'
        self.network_type_dict_inv[14] = 'WASTEMANAGEMENT'
        self.network_type_dict_inv[15] = 'EDUCATION'
        self.network_type_dict_inv[16] = 'HEALTHCARE'
        self.network_type_dict_inv[17] = 'ART'
        self.network_type_dict_inv[18] = 'FOOD'
        self.network_type_dict_inv[19] = 'OTHER'
        self.network_type_dict_inv[20] = 'CHILD'
        self.network_type_dict_inv[21] = 'ELDERLY'
        self.network_type_dict_inv[22] = 'random'

        self.agents_mean_interactions = 0*torch.ones(self.params['num_agents'], self.num_networks).to(self.device) #Age and Network and Occupation may need to be checked to populate this
        mean_int_households_df = pd.read_csv('Data/GeneratedAgentsData/HouseholdsData.csv')
        mean_int_households_len = torch.tensor(mean_int_households_df['Members'].apply(lambda x: len(x)).values).long().to(self.device)
        mean_int_occ_df = pd.read_csv('Data/Initialization/OccupationNetworkParameters.csv')
        mean_int_occ_mu = torch.tensor(mean_int_occ_df['mu'].values).float().to(self.device)
        mean_int_ran_df = pd.read_csv('Data/Initialization/RandomNetworkParameters.csv')
        mean_int_ran_mu = torch.tensor(mean_int_ran_df['mu'].values).float().to(self.device)

        mean_int_ran_sigma = torch.tensor(mean_int_ran_df['sigma'].values).float().to(self.device)

        self.agents_random_interactions = torch.tensor([
            get_num_random_interactions(age, mean_int_ran_mu, mean_int_ran_sigma, self.params['CHILD_Upper_Index'],
                                        self.params['ADULT_Upper_Index'])
            for age in self.agents_ages
        ])

        #For household mean interactions:
        self.agents_mean_interactions[:,0] = mean_int_households_len[self.agents_households.long()]
        for occ in self.params['occupations_ix_list']:
            self.agents_mean_interactions[:,occ+1] = mean_int_occ_mu[occ] #since forst index is used for household
        child_agents = self.agents_ages <= self.params['CHILD_Upper_Index']
        adult_agents = torch.logical_and(self.agents_ages > self.params['CHILD_Upper_Index'], self.agents_ages <= self.params['ADULT_Upper_Index']).view(-1)
        elderly_agents = self.agents_ages > self.params['ADULT_Upper_Index']
        self.agents_mean_interactions[child_agents.bool(), 22] = mean_int_ran_mu[0]
        self.agents_mean_interactions[adult_agents.bool(), 22] = mean_int_ran_mu[1]
        self.agents_mean_interactions[elderly_agents.bool(), 22] = mean_int_ran_mu[2]
        self.agents_mean_interactions_split = list(torch.split(self.agents_mean_interactions, 1, dim=1))
        self.agents_mean_interactions_split = [a.view(-1) for a in self.agents_mean_interactions_split]
        self.R = 5.02 #5.18
        self.SFSusceptibility  = torch.tensor([ 0.35, 0.69, 1.03, 1.03, 1.03, 1.03, 1.27, 1.52, 1.52]).float().to(self.device)
        self.SFInfector = torch.tensor([0.0, 0.33, 0.05, 0.05, 0.72, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).float().to(self.device)

        self.B_n = {}
        self.B_n['household'] = 2
        self.B_n['occupation'] = 1
        self.B_n['random'] = 0.25
        self.lam_gamma = {}
        self.lam_gamma['mu'] = 5.5
        self.lam_gamma['sigma'] = 2.14
        self.lam_gamma_integrals = self._get_lam_gamma_integrals(**self.lam_gamma, t = self.params['num_steps'])
        self.net = InfectionNetwork(lam, self.R, self.SFSusceptibility, 
                                    self.SFInfector, self.lam_gamma_integrals).to(self.device) #Initializing message passing network (currently no trainable parameters here)
        #**********************************************************************************
        #Dynamic
        self.den_contacts = deque([], self.params['max_den_contact_days'])
        #**********************************************************************************
        #**********************************************************************************
        #Agent state variables
        #**********************************************************************************
        # infile = os.path.join(
        #         self.params['output_location']['parent_dir'],
        #         self.params['output_location']['agents_dir'],
        #         self.params['output_location']['agents_outfile'])
        # agents_df = pd.read_csv(infile)
        #**********************************************************************************
        #Static
        # self.agents_ages = torch.tensor(agents_df['age_group'].to_numpy()).long().to(self.device) #Assuming age remains constant in the simulation
        self.agents_occupation = torch.tensor(agents_df['occupation_network'].to_numpy()).long().to(self.device) #Assuming occupation remains constant in the simulation
        self.agents_app_users = torch.tensor(agents_df['app_user'].to_numpy()).long().to(self.device) #Assuming app user status remains constant in the simulation
        self.agents_app_users = torch.logical_and(self.agents_app_users.bool(), torch.distributions.Categorical(probs=torch.tensor([1-self.params['app_on_prob'], self.params['app_on_prob']])).sample((self.params['num_agents'],)).bool() ).view(-1)

        #**********************************************************************************
        #Dynamic
        #a.Testing
        self.agents_test_eligibility = torch.ones(self.params['num_steps']+1, self.params['num_agents']).long().to(self.device)
        self.agents_test_results = torch.zeros(self.params['num_steps']+1, self.params['num_agents']).long().to(self.device)
        self.agents_awaiting_test_results = torch.zeros(self.params['num_agents']).long().to(self.device)
        self.agents_test_results_dates = -1*torch.ones(self.params['num_steps']+1, self.params['num_agents']).to(self.device)
        self.last_test_date = -1*torch.zeros(self.params['num_agents']).to(self.device)

        self.den_test_eligibility = torch.ones(self.params['num_agents']).long().to(self.device)
        if self.params['use_poc_test_logic']:
            self.den_test_eligibility = torch.zeros(self.params['num_agents']).long().to(self.device)
        else:
            self.den_test_eligibility = torch.ones(self.params['num_agents']).long().to(self.device)
        self.agents_poc_test_results = torch.zeros(self.params['num_steps']+1, self.params['num_agents']).long().to(self.device)
        self.last_poc_test_date = -1*torch.zeros(self.params['num_agents']).to(self.device)
        #b.Quarantine
        self.is_quarantined = torch.zeros(self.params['num_steps'], self.params['num_agents']).to(self.device)
        self.quarantine_start_date = (self.params['num_steps']+1)*torch.ones(self.params['num_agents']).to(self.device)
        #c.Infection and Disease
        self.agents_stages = torch.stack([torch.tensor(agents_df['stage'].to_numpy()).long()] + 
                [-1*torch.ones_like(torch.tensor(agents_df['stage'].to_numpy()))]*(params['num_steps'])).to(self.device) #Initialized for num_steps + 1
        self.agents_infected_index = (self.agents_stages > 0).to(self.device) #Not susceptible
        self.agents_infected_time = ((self.params['num_steps']+1)*torch.ones_like(self.agents_stages)).to(self.device) #Practically infinite as np.inf gives wrong data type
        self.agents_infected_time[0, self.agents_infected_index[0].bool()] = 0
        self.agents_next_stages = -1*torch.ones_like(self.agents_stages[0]).to(self.device)
        self.agents_next_stage_times = (self.params['num_steps']+1)*torch.ones_like(self.agents_stages[0]).to(self.device).long() #Practically infinite as np.inf gives wrong data type
        #c. Vaccination
        self.vaccine_availability = torch.zeros(self.params['num_steps']).long().to(self.device)
        self.vaccine_availability[self.params['vaccine_start_date']:] = self.params['vaccine_daily_production']
        self.agents_vaccine_no_doses = torch.ones(self.params['num_agents']).long().to(self.device)
        self.agents_vaccine_current_one_dose = torch.zeros(self.params['num_agents']).long().to(self.device)
        self.agents_vaccine_current_two_doses = torch.zeros(self.params['num_agents']).long().to(self.device)
        self.agents_first_dose_date = (self.params['num_steps']+1)*torch.ones(self.params['num_agents']).long().to(self.device)
        self.agents_second_dose_date = (self.params['num_steps']+1)*torch.ones(self.params['num_agents']).long().to(self.device)

        self.agents_vaccination_effectiveness = torch.zeros(self.params['num_steps'] + 1,
                                                            self.params['num_agents']).long().to(self.device)
        #**********************************************************************************
        #Household network creation
        #Forward and backward edges need to be added as by default the message passing network is directional
        infile = os.path.join(get_dir_from_path_list(
                [params['output_location']['parent_dir'],
                    params['output_location']['networks_dir'],
                    params['output_location']['household_networks_dir']]
        ), 'household.csv')
        household_network_edgelist_forward = torch.tensor(pd.read_csv(infile, header=None).to_numpy()).t().long().to(self.device)
        household_network_edgelist_backward = torch.vstack((household_network_edgelist_forward[1,:], household_network_edgelist_forward[0,:]))
        self.household_network_edgelist = torch.hstack((household_network_edgelist_forward, household_network_edgelist_backward))
        #Type needs to be obtained from a variable. This same type needs to be used in B_n
        household_network_edgeattr_type = self.network_type_dict['household']*torch.ones(self.household_network_edgelist.shape[1]).long().to(self.device)
        household_network_edgeattr_B_n = (torch.ones(self.household_network_edgelist.shape[1]).float()*self.B_n['household']).to(self.device)
        self.household_network_edgeattr = torch.vstack((household_network_edgeattr_type, household_network_edgeattr_B_n))
        #**********************************************************************************
        #Reading disease stage dynamics data
        infile = os.path.join(get_dir_from_path_list(
                [params['output_location']['parent_dir'],
                    params['output_location']['dynamics_dir'],
                    ]
        ), params['output_location']['disease_dynamics_stages_file_name'])
        disease_df = pd.read_csv(infile, index_col=[0,1]).fillna(0)
        self.disease_probs_age = torch.tensor(disease_df.to_numpy()).view(-1, params['num_stages'], params['num_stages']).to(self.device)
        #Reading disease transition times dynamics data
        infile = os.path.join(get_dir_from_path_list(
                [params['output_location']['parent_dir'],
                    params['output_location']['dynamics_dir'],
                    ]
        ), params['output_location']['disease_dynamics_transition_times_file_name'])
        disease_transition_times_df = pd.read_csv(infile, index_col=[0,1]).fillna(0)
        disease_transition_times_dists_df = disease_transition_times_df.copy()
        disease_transition_times_mu_df = disease_transition_times_df.copy()
        disease_transition_times_sigma_df = disease_transition_times_df.copy()
        for col in disease_transition_times_df.columns:
            disease_transition_times_dists_df[col] = disease_transition_times_df[col].apply(time_dists_split(0, int))
            disease_transition_times_mu_df[col] = disease_transition_times_df[col].apply(time_dists_split(1, np.float))
            disease_transition_times_sigma_df[col] = disease_transition_times_df[col].apply(time_dists_split(2, np.float))
        self.disease_transition_times_dists_age = torch.tensor(disease_transition_times_dists_df.to_numpy()).view(-1, self.params['num_stages'], self.params['num_stages']).to(self.device)
        self.disease_transition_times_mu_age = torch.tensor(disease_transition_times_mu_df.to_numpy()).view(-1, self.params['num_stages'], self.params['num_stages']).to(self.device)
        self.disease_transition_times_sigma_age = torch.tensor(disease_transition_times_sigma_df.to_numpy()).view(-1, self.params['num_stages'], self.params['num_stages']).to(self.device)
        #**********************************************************************************
        #Setting the next stages and next stage transition times for the initialy infected agents
        self._set_next_stages_times(self.agents_infected_index[0,:], self.agents_stages[0, :], 0)
        #**********************************************************************************
        self.current_time = 0


    def step(self):
        t = self.current_time
        print('*'*60)
        print("This is time {}".format(t))
        '''
        Timing diagram:
        Agent first updates test results
        Test semi-dynamic: test_results_date, tested_awaiting_results (dynamic and overwritten)

        Test dynamic state: tested, tested_got_result_today, test_positive
        if not tested:
            check for symptomatic states (with certainty) or den (with den_comply probability)
            if gone for testing now:
                update tested_today should be made 1
                test_results_date should be filled
                tested_awaiting results should be made 1
        else:
            if tested_got_result_today: (set this value)
            sample test result - based on agent disease stage and test accuracy
            if positive then quarantine - will always go initially, then there is a drop probability below
            i.e. set quaratine start day and quarantine end day (as start day + 14)
        
        Quarantine dynamics:
        StateL Quarantine start day, quarantine end day, Is_Quarantined - all these are dynamic binary variables
        Check for completion of quarantine - cahnge is_quarantined
        With a certain drop probability, agents in quarantine stay or drop
        if agent drops quarantine:
            quarantine end day will be today
            is_quarantined to be set to 0

        DEN dynamics
        maintain a list of edges for last 7 days
        only agents that received a positive test result today will send DEN
        with current infected send message to agents contacted in last 7 days if both are app users
        '''
        #**********************************************************************************
        #1. Test dynamics #TODO change the order for 1a and 1b to account for same day test results
        #----------------------------------------------------------------------------------
        #1a. Agents get test results
        #self.agents_test_results_date[t,:] is filled with the day the test was taken
        agents_test_results_expected_today = self.agents_test_results_dates[t,:] >= 0
        self.agents_awaiting_test_results[agents_test_results_expected_today.bool()] = 0
        if self.params['test_validity_days'] > 0:
            self.agents_test_eligibility[t:t+self.params['test_validity_days'], agents_test_results_expected_today.bool()] = 0
        if self.params['debug']:
            print('There are {} agents awaiting test results today'.format(agents_test_results_expected_today.sum()))
        test_dates_of_agents_test_results_expected_today = self.agents_test_results_dates[t,:].clamp(min=0)
        ix = torch.vstack((test_dates_of_agents_test_results_expected_today, torch.arange(0, self.params['num_agents']))).long()
        test_stages_of_agents_test_results_expected_today = self.agents_stages[ix.chunk(chunks=2, dim=0)]
        positive_result_candidates = torch.logical_and(
            torch.logical_and(test_stages_of_agents_test_results_expected_today > 0, 
                test_stages_of_agents_test_results_expected_today != 10), 
                    test_stages_of_agents_test_results_expected_today != 8).view(-1)
        negative_result_candidates = torch.logical_not(positive_result_candidates)
        positive_test_candidates = torch.logical_and(positive_result_candidates, agents_test_results_expected_today).view(-1)
        negative_test_candidates = torch.logical_and(negative_result_candidates, agents_test_results_expected_today).view(-1)
        if self.params['debug']:
            print('There are {} positive candidates for test results today'.format(positive_test_candidates.sum()))
            print('There are {} negative candidates for test results today'.format(negative_test_candidates.sum()))
        if positive_test_candidates.sum() > 0:
            positive_test_results = torch.distributions.Categorical(
                probs=torch.tensor([1-self.params['test_true_positive'], self.params['test_true_positive']])
                ).sample((torch.sum(positive_test_candidates),))
            self.agents_test_results[t,positive_test_candidates.bool()] = positive_test_results
            if self.params['debug']:
                print('Of the positive candidates for test results today {} have tested positive'.format(positive_test_results.sum()))
        if negative_test_candidates.sum() > 0:
            negative_test_results = torch.distributions.Categorical(
                probs=torch.tensor([1-self.params['test_false_positive'], self.params['test_false_positive']])
                ).sample((torch.sum(negative_test_candidates),))
            self.agents_test_results[t,negative_test_candidates.bool()] = negative_test_results
            if self.params['debug']:
                print('Of the negative candidates for test results today {} have tested positive'.format(negative_test_results.sum()))
        #----------------------------------------------------------------------------------
        #1b. Agents get themselves tested
        den_test_recommended = torch.zeros(self.params['num_agents']).long().to(self.device)
        for d, edges_d in enumerate(self.den_contacts):
            den_contact_day = t - len(self.den_contacts) + d
            test_not_yet_done = self.last_test_date < den_contact_day
            adj = SparseTensor(row=edges_d[0], col=edges_d[1], 
                        sparse_sizes=(self.params['num_agents'], self.params['num_agents'])).to(self.device)
            infectious_neighbors_with_app = torch.logical_and(self.agents_test_results[t,:], self.agents_app_users) #Line which justifies current order
            infectious_neighbors_with_app = torch.logical_and(infectious_neighbors_with_app, torch.logical_not(self.is_quarantined[den_contact_day,:])) #Line which justifies current order
            positive_contacts = adj.matmul(infectious_neighbors_with_app.view(-1,1).long()).view(-1)
            den_test_recommended[torch.logical_and(positive_contacts, test_not_yet_done).bool()] = 1
            if t > 0: #Addiing logic for poc test contacts sending out with DEN
                poc_infected_will_inform_den = torch.distributions.Categorical(probs = torch.tensor([1-self.params['poc_den_inform_prob'], self.params['poc_den_inform_prob']])).sample((self.params['num_agents'],)).bool().to(self.device)
                poc_infectious_neighbors_with_app = torch.logical_and(self.agents_poc_test_results[t-1,:].bool(), self.agents_app_users) #Line which justifies current order
                poc_infectious_neighbors_with_app = torch.logical_and(poc_infectious_neighbors_with_app, poc_infected_will_inform_den).view(-1).bool()
                poc_infectious_neighbors_with_app = torch.logical_and(poc_infectious_neighbors_with_app, torch.logical_not(self.is_quarantined[den_contact_day,:])) #Line which justifies current order
                poc_positive_contacts = adj.matmul(poc_infectious_neighbors_with_app.view(-1,1).long()).view(-1)
                den_test_recommended[torch.logical_and(poc_positive_contacts, test_not_yet_done).bool()] = 1
        den_test_recommended = torch.logical_and(den_test_recommended, self.agents_app_users)

        #Adding poc test logic here: - test, get results
        if (self.params['use_poc_test_logic'] and (t >= self.params['poc_test_start_date'])):
            if self.params['poc_test_only_for_den']:
                poc_test_candidates = torch.clone(den_test_recommended)
            else:
                poc_test_candidates = torch.ones(self.params['num_agents']).bool().to(self.device)
                poc_test_candidates = torch.logical_and(poc_test_candidates, self.agents_app_users.bool()).view(-1)
            poc_test_candidates = torch.logical_and(poc_test_candidates, 
            torch.distributions.Categorical(probs = 
            torch.tensor([1-self.params['poc_acceptance_prob'], self.params['poc_acceptance_prob']])).sample((self.params['num_agents'],)).bool()
            ).view(-1).bool()
            if not(self.params['poc_test_on_symptoms']):
                symptomatic_cases = torch.logical_or(self.agents_stages[t,:] == 4, self.agents_stages[t,:] == 5) #Mild or Severe symptoms
                poc_test_candidates = torch.logical_and(poc_test_candidates, torch.logical_not(symptomatic_cases)).view(-1).bool()
            poc_test_candidates = torch.logical_and(poc_test_candidates, torch.logical_not(self.is_quarantined[t,:])).view(-1)
            poc_test_eligible_stage_agents = self.agents_stages[t,:] <= 5
            poc_test_candidates = torch.logical_and(poc_test_candidates, poc_test_eligible_stage_agents).view(-1)
            if self.params['debug']:
                print('There are {} agents for the POC test today'.format(poc_test_candidates.sum()))
            poc_positive_result_candidates = torch.logical_and(
                torch.logical_and(self.agents_stages[t, :] > 0, self.agents_stages[t, :] != 10), 
                self.agents_stages[t, :] != 8).view(-1)
            poc_negative_result_candidates = torch.logical_not(poc_positive_result_candidates)
            poc_positive_result_candidates = torch.logical_and(poc_positive_result_candidates, poc_test_candidates)
            poc_negative_result_candidates = torch.logical_and(poc_negative_result_candidates, poc_test_candidates)
            if poc_positive_result_candidates.sum() > 0:
                poc_positive_test_results = torch.distributions.Categorical(
                    probs=torch.tensor([1-self.params['poc_test_true_positive'], self.params['poc_test_true_positive']])
                    ).sample((torch.sum(poc_positive_result_candidates),))
                self.agents_poc_test_results[t,poc_positive_result_candidates.bool()] = poc_positive_test_results
                self.last_poc_test_date[poc_positive_result_candidates.bool()] = t
            if poc_negative_result_candidates.sum() > 0:
                poc_negative_test_results = torch.distributions.Categorical(
                    probs=torch.tensor([1-self.params['poc_test_false_positive'], self.params['poc_test_false_positive']])
                    ).sample((torch.sum(poc_negative_result_candidates),))
                self.agents_poc_test_results[t,poc_negative_result_candidates.bool()] = poc_negative_test_results
                self.last_poc_test_date[poc_negative_result_candidates.bool()] = t
            if self.params['debug']:
                print('There are {} positive candidates for test results today'.format(poc_positive_result_candidates.sum()))
                if poc_positive_result_candidates.sum() > 0:
                    print('Of the positive candidates for test results today {} have tested positive'.format(poc_positive_test_results.sum()))
                print('There are {} negative candidates for test results today'.format(poc_negative_result_candidates.sum()))
                if poc_negative_result_candidates.sum() > 0:
                    print('Of the negative candidates for test results today {} have tested positive'.format(poc_negative_test_results.sum()))
            
        #Continuing non POC test logic
        den_test_recommended = torch.logical_and(self.den_test_eligibility.bool(), den_test_recommended.bool())
        if self.params['debug']:
            print('There are {} agents who have received DEN today'.format(den_test_recommended.sum()))
        den_test_will_comply = torch.distributions.Categorical(
            probs=torch.tensor([1-self.params['den_will_comply'], self.params['den_will_comply']])).sample((self.params['num_agents'],))
        den_tests = self.params['use_den_logic']*torch.logical_and(den_test_recommended, den_test_will_comply)
        if self.params['debug']:
            print('Of these agents only {} will comply with DEN today'.format(den_tests.sum()))
        symptomatic_cases = torch.logical_or(self.agents_stages[t,:] == 4, self.agents_stages[t,:] == 5) #Mild or Severe symptoms
        self.agents_test_eligibility[t,symptomatic_cases.bool()] = 1 #Added on Dec 30 night #TODO
        if self.params['debug']:
            print('There are {} agents who have mild or sever symptoms today'.format(symptomatic_cases.sum()))
        will_take_tests = torch.logical_or(symptomatic_cases, den_tests).view(-1)
        will_take_tests = torch.logical_and(will_take_tests, torch.logical_not(self.is_quarantined[t,:])).view(-1)
        if self.params['debug']:
            print('There are finally {} agents who are supposed to take tests today'.format(will_take_tests.sum()))
        will_take_tests = torch.logical_and(will_take_tests.bool(), self.agents_test_eligibility[t,:].bool())
        will_take_tests = torch.logical_and(will_take_tests.bool(), torch.logical_not(self.agents_awaiting_test_results.bool()))
        test_eligible_stage_agents =  self.agents_stages[t,:] <= 5
        will_take_tests = torch.logical_and(will_take_tests, test_eligible_stage_agents).view(-1)
        self.agents_awaiting_test_results[will_take_tests.bool()] = 1
        if self.params['debug']:
            print('Of the agents who are supposed to take tests only {} agents will take tests today as others are awaiting test results'.format(will_take_tests.sum()))
        self.last_test_date[will_take_tests.bool()] = t
        if will_take_tests.sum() > 0:
            test_results_dates_ix = torch.distributions.Categorical(probs=torch.tensor(self.params['test_results_dates_probs'])).sample((will_take_tests.sum(),))
            test_results_dates = torch.clamp(t + torch.tensor(self.params['test_results_dates'])[test_results_dates_ix.long()], max = self.params['num_steps'])
            ix = torch.vstack((test_results_dates, torch.arange(0, self.params['num_agents'])[will_take_tests.bool()]))
            self.agents_test_results_dates[ix.chunk(chunks=2, dim=0)] = t 
        #----------------------------------------------------------------------------------
        #**********************************************************************************
        #2. Quarantine dynamics
        #----------------------------------------------------------------------------------
        #2a. End quarantine
        agents_quarantine_end_date = self.quarantine_start_date + self.params['quarantine_days']
        agents_quarantine_ends = t >= agents_quarantine_end_date
        if self.params['debug']:
            print('There are {} agents whose quarantine ends today'.format(agents_quarantine_ends.sum()))
        if agents_quarantine_ends.sum() > 0:
            self.is_quarantined[t, agents_quarantine_ends.bool()] = 0
            self.quarantine_start_date[agents_quarantine_ends.bool()] = self.params['num_steps']+1
        #----------------------------------------------------------------------------------
        #2b. Add to quarantine from test
        if self.params['debug']:
            print('All of the {} positively tested agents begin a {} day quarantine'.format(self.agents_test_results[t,:].bool().sum(), self.params['quarantine_days']))
        if self.agents_test_results[t,:].bool().sum() > 0:
            self.quarantine_start_date[self.agents_test_results[t,:].bool()] = t
            self.is_quarantined[t, self.agents_test_results[t,:].bool()] = 1
        if self.params['debug']:
            print('At this point, before dropping off, {} agents are in quarantine'.format(self.is_quarantined[t,:].sum()))
        #Adding poc quarantine logic
        poc_agents_will_quarantine = torch.logical_and(
            self.agents_poc_test_results[t,:].bool(),
            torch.distributions.Categorical(
                probs = torch.tensor([1-self.params['poc_quarantine_enter_prob'], self.params['poc_quarantine_enter_prob']])
            ).sample((self.params['num_agents'],)).bool() 
        ).view(-1).bool()
        if self.params['debug']:
            print('Of the {} POC positively tested agents, {} agents begin a {} day quarantine'.format(self.agents_poc_test_results[t,:].bool().sum(), poc_agents_will_quarantine.bool().sum(), self.params['quarantine_days']))
        if poc_agents_will_quarantine.bool().sum() > 0:
            self.quarantine_start_date[poc_agents_will_quarantine.bool()] = t
            self.is_quarantined[t, poc_agents_will_quarantine.bool()] = 1
        if self.params['debug']:
            print('At this point, before dropping off, {} agents are in quarantine including {} agents from POC test'.format(self.is_quarantined[t,:].sum(), poc_agents_will_quarantine.sum()))
        #----------------------------------------------------------------------------------
        #2c. Break quarantine
        agents_breaking_quarantine = torch.distributions.Categorical(
            probs=torch.tensor([1-self.params['quarantine_break_prob'], 
            self.params['quarantine_break_prob']])).sample((self.params['num_agents'],))
        agents_breaking_quarantine = torch.logical_and(self.is_quarantined[t], agents_breaking_quarantine)
        if self.params['debug']:
            print('{} agents are breaking quarantine today'.format(agents_breaking_quarantine.sum()))
        if agents_breaking_quarantine.sum() > 0:
            self.is_quarantined[t, agents_breaking_quarantine.bool()] = 0
            self.quarantine_start_date[agents_breaking_quarantine.bool()] = self.params['num_steps']+1
        if self.params['debug']:
            print('At this point, after dropping off, {} agents are in quarantine'.format(self.is_quarantined[t,:].sum()))
        #----------------------------------------------------------------------------------
        #**********************************************************************************
        #3. Vaccination dynamics
        #----------------------------------------------------------------------------------
        #3a. Before vaccination
        if self.params['use_vaccination_logic']:
            if self.params['debug']:
                print('Using vaccination logic now at time {}'.format(t))
            #Prioritization - ages + dosage
            agents_receiving_vaccine = torch.zeros(self.params['num_agents']).long().to(self.device)
            #TODO
            prioritization_vector = (self.agents_ages * 1 + 
                                ((self.params['vaccine_first_dose_priority'])*self.agents_vaccine_no_doses + (1-self.params['vaccine_first_dose_priority'])*self.agents_vaccine_current_one_dose)*100 + 
                                ((1-self.params['vaccine_first_dose_priority'])*self.agents_vaccine_no_doses + (self.params['vaccine_first_dose_priority'])*self.agents_vaccine_current_one_dose)*10)
            _, agent_vaccination_order = torch.sort(prioritization_vector, descending=True)
            # agent_vaccination_order = agent_vaccination_order.view(-1)
            vaccine_drop_probs = torch.zeros(self.params['num_agents']).float().to(self.device)
            vaccine_drop_probs[self.agents_vaccine_current_one_dose.bool()] = self.params['vaccine_drop_prob_before_second_dose']
            will_take_vaccine = torch.distributions.Categorical(probs = torch.vstack((vaccine_drop_probs, 1-vaccine_drop_probs)).t()).sample().long()
            #Do not vaccinate if tested positive or hospitalized
            vaccination_ineligible = torch.logical_or(self.agents_stages[t,:] >= 6, self.is_quarantined[t,:].bool()).view(-1)
            will_take_vaccine = torch.logical_and(will_take_vaccine, torch.logical_not(vaccination_ineligible)).view(-1)
            #Do not vaccinate if delay time has not passed for second dose
            agents_not_yet_eligible_for_second_dose = torch.logical_and(self.agents_vaccine_current_one_dose.bool(), 
                    (self.agents_first_dose_date + self.params['vaccine_second_dose_delay'] > t))
            will_take_vaccine = torch.logical_and(will_take_vaccine, torch.logical_not(agents_not_yet_eligible_for_second_dose)).view(-1)
            will_take_vaccine = torch.logical_and(will_take_vaccine, torch.logical_not(self.agents_vaccine_current_two_doses.bool())).view(-1)
            will_take_vaccine_in_priority_order = will_take_vaccine[agent_vaccination_order.long()]
            vaccine_available_today = self.vaccine_availability[max(t-self.params['vaccine_shelf_life'], self.params['vaccine_start_date']):t+1].sum()
            #3 cases
            if will_take_vaccine_in_priority_order.sum() == 0:
                vaccine_greater_than_agents = True
                agent_index_upto_which_vaccine_available = 0
            elif vaccine_available_today > will_take_vaccine_in_priority_order.sum(): #More vaccine than agents
                vaccine_greater_than_agents = True
                agent_index_upto_which_vaccine_available = len(will_take_vaccine_in_priority_order)
            elif vaccine_available_today == 0:
                vaccine_greater_than_agents = False
                agent_index_upto_which_vaccine_available = 0
            else:
                vaccine_greater_than_agents = False
                agent_index_upto_which_vaccine_available = torch.argmax((torch.cumsum(will_take_vaccine_in_priority_order, 0).long() == vaccine_available_today).long()) + 1
            agents_ix_vaccine_available = agent_vaccination_order[:agent_index_upto_which_vaccine_available]
            agents_receiving_vaccine[agents_ix_vaccine_available.long()] = 1
            agents_receiving_vaccine = torch.logical_and(agents_receiving_vaccine.bool(), will_take_vaccine.bool())
            # agents_receiving_vaccine = torch.clone(will_take_vaccine_in_priority_order).long()
            # agents_receiving_vaccine[agent_index_upto_which_vaccine_available+1:] = 0
        
            if self.params['debug']:
                print('Vaccine available today is {} units'.format(vaccine_available_today))
                print('{} agents are eligible for vaccine today'.format(will_take_vaccine_in_priority_order.sum()))
                print('More vaccine than agents is {}'.format(vaccine_greater_than_agents))
                print('{} agents are receiving vaccine today'.format(agents_receiving_vaccine.sum()))
            #----------------------------------------------------------------------------------
            #3b. During Vaccination 
            #Giving second dose
            agents_getting_second_dose = torch.logical_and(agents_receiving_vaccine.bool(), self.agents_vaccine_current_one_dose.bool()).view(-1).bool() #TODO
            self.agents_vaccine_current_one_dose[agents_getting_second_dose.bool()] = 0
            self.agents_vaccine_current_two_doses[agents_getting_second_dose.bool()] = 1
            self.agents_second_dose_date[agents_getting_second_dose.bool()] = t


            if agents_getting_second_dose.sum() > 0:
                self.agents_vaccination_effectiveness[t:,agents_getting_second_dose.bool()] = torch.distributions.Categorical(probs=torch.tensor([1 - self.params['vaccine_second_dose_effectiveness'],
                     self.params['vaccine_second_dose_effectiveness']])).sample((agents_getting_second_dose.bool().sum(),)).long()

                self.agents_random_interactions[agents_getting_second_dose.bool()] = self.params['scale_random_interact'] * self.agents_random_interactions[agents_getting_second_dose.bool()]

            if self.params['debug']:
                print("Scaled random interactions by: ", self.params['scale_random_interact'])

            #Giving first dose
            agents_getting_first_dose = torch.logical_and(agents_receiving_vaccine.bool(), self.agents_vaccine_no_doses.bool()).view(-1).bool()
            self.agents_vaccine_no_doses[agents_getting_first_dose.bool()] = 0
            self.agents_vaccine_current_one_dose[agents_getting_first_dose.bool()] = 1
            self.agents_first_dose_date[agents_getting_first_dose.bool()] = t

            if agents_getting_first_dose.sum() > 0:
                self.agents_vaccination_effectiveness[min(t + self.params['vaccine_first_dose_kick_in_days'], self.params['num_steps']):, agents_getting_first_dose.bool()] = torch.distributions.Categorical(
                    probs=torch.tensor([1 - self.params['vaccine_first_dose_effectivness'], self.params['vaccine_first_dose_effectivness']])).sample((agents_getting_first_dose.bool().sum(),)).long()

                self.agents_random_interactions[agents_getting_first_dose.bool()] = self.params['scale_random_interact']*self.agents_random_interactions[agents_getting_first_dose.bool()]


            if self.params['debug']:
                print('{} agents are getting first dose today'.format(agents_getting_first_dose.sum()))
                print('{} agents are getting second dose today'.format(agents_getting_second_dose.sum()))
                print('agents vaccine status over days {}'.format(self.agents_vaccination_effectiveness[t,:].sum()))
            #Removing from vaccine availability
            if vaccine_greater_than_agents:
                vaccine_doses_consumed = will_take_vaccine_in_priority_order.sum()
                vaccine_doses_remaining_to_be_removed = vaccine_doses_consumed
                for tau in range(max(t-self.params['vaccine_shelf_life'], self.params['vaccine_start_date']),t+1):
                    if vaccine_doses_remaining_to_be_removed > self.vaccine_availability[tau]:
                        vaccine_doses_remaining_to_be_removed -= self.vaccine_availability[tau]
                        self.vaccine_availability[tau] = 0
                    else:
                        self.vaccine_availability[tau] -= vaccine_doses_remaining_to_be_removed
                        vaccine_doses_remaining_to_be_removed = 0
                    if vaccine_doses_remaining_to_be_removed == 0:
                        break
            else:
                vaccine_doses_consumed = vaccine_available_today
                self.vaccine_availability[max(t-self.params['vaccine_shelf_life'], self.params['vaccine_start_date']):t+1] = 0
            #Remove quarantine if vaccinated (edge cases)
            self.is_quarantined[t,agents_getting_first_dose] = 0
            self.quarantine_start_date[agents_getting_first_dose] = self.params['num_steps']+1
            self.is_quarantined[t,agents_getting_second_dose] = 0
            self.quarantine_start_date[agents_getting_second_dose] = self.params['num_steps']+1
            #----------------------------------------------------------------------------------
            #3c. After vaccination
            #Computing vaccine effectiveness
            #Change in vaccine effectiveness for people with current one dose
            # if self.agents_vaccine_current_one_dose.sum() > 0:
            #     self.agents_vaccination_effectiveness[self.agents_vaccine_current_one_dose.bool()] = (
            #         0*(t < (self.agents_first_dose_date[self.agents_vaccine_current_one_dose.bool()] +
            #                     self.params['vaccine_first_dose_kick_in_days'])) +
            #         self.params['vaccine_first_dose_effectivness']*(t >= (self.agents_first_dose_date[self.agents_vaccine_current_one_dose.bool()] +
            #                     self.params['vaccine_first_dose_kick_in_days']) )
            #     ).float()
            # # Change in vaccine effectiveness for people with current two doses
            # self.agents_vaccination_effectiveness[self.agents_vaccine_current_two_doses.bool()] = self.params['vaccine_second_dose_effectiveness']
            # self.agents_vaccination_effectiveness[(self.agents_stages[t,:] != 0).bool()] = 0 #Not really needed

            if self.params['debug']:
                print('As of today {} agents are yet to be vaccinated'.format(self.agents_vaccine_no_doses.sum()))
                print('As of today {} agents have received just one dose of vaccine'.format(self.agents_vaccine_current_one_dose.sum()))
                print('As of today {} agents have received two doses of vaccine'.format(self.agents_vaccine_current_two_doses.sum()))
                print('Vaccine available prior to today\'s vaccination is {} units'.format(vaccine_available_today))
                vaccine_available_after_today = self.vaccine_availability[max(t-self.params['vaccine_shelf_life'], self.params['vaccine_start_date']):t+1].sum()
                print('Vaccine available after today\'s vaccination is {} units'.format(vaccine_available_after_today))
        #----------------------------------------------------------------------------------
        #**********************************************************************************
        #4. Infection dynamics
        #Initializing values to same as last time
        self.agents_stages[t+1, :] = self.agents_stages[t,:]
        self.agents_infected_index[t+1,:] = self.agents_infected_index[t,:]
        self.agents_infected_time[t+1, :] = self.agents_infected_time[t, :]
        # self.is_quarantined[t,:] = 0
        # self.agents_vaccination_effectiveness[:] = 1
        #Preparing agents states for message passing network at the current time
        x = torch.stack((
            self.agents_ages,  #0
            self.agents_stages[t],  #1 
            self.agents_infected_index[t], #2
            self.agents_infected_time[t], #3
            *self.agents_mean_interactions_split, #4 to 26
            torch.arange(self.params['num_agents']).to(self.device), #Agent ids (27)
            self.params['use_quarantine_logic']*self.is_quarantined[t],     #28
            self.agents_vaccination_effectiveness[t,:]       #29
    )).t()
        #Reading Occupation Network edges
        all_occupation_network_edgelist = []
        all_occupation_network_edgeattr = []
        for occupation in range(1, 22):
            try:
                infile = os.path.join(get_dir_from_path_list(
                    [self.params['output_location']['parent_dir'],
                        self.params['output_location']['networks_dir'],
                        self.params['output_location']['occupation_networks_dir'], 
                        self.network_type_dict_inv[occupation]]
                        ), '{}.csv'.format(t))
                df = pd.read_csv(infile, header=None)
            except:
                print('People of occupation {} not present in data'.format(self.network_type_dict_inv[occupation]))
                continue
            occupation_network_edgelist_forward = torch.tensor(df.to_numpy()).t().long()
            occupation_network_edgelist_backward = torch.vstack((occupation_network_edgelist_forward[1,:], occupation_network_edgelist_forward[0,:]))
            occupation_network_edgelist = torch.hstack((occupation_network_edgelist_forward, occupation_network_edgelist_backward))
            occupation_network_edgeattr_type = occupation*torch.ones(occupation_network_edgelist.shape[1]).long()
            occupation_network_edgeattr_B_n = torch.ones(occupation_network_edgelist.shape[1]).float()*self.B_n['occupation']
            occupation_network_edgeattr = torch.vstack((occupation_network_edgeattr_type, occupation_network_edgeattr_B_n))
            all_occupation_network_edgelist.append(occupation_network_edgelist)
            all_occupation_network_edgeattr.append(occupation_network_edgeattr)

        all_occupation_network_edgelist = torch.hstack(all_occupation_network_edgelist)
        all_occupation_network_edgeattr = torch.hstack(all_occupation_network_edgeattr)

        #Reading Random Network edges
        # infile = os.path.join(get_dir_from_path_list(
        #             [self.params['output_location']['parent_dir'],
        #                 self.params['output_location']['networks_dir'],
        #                 self.params['output_location']['random_networks_dir']]
        #                 ), '{}.csv'.format(t))
        # random_network_edgelist_forward = torch.tensor(pd.read_csv(infile, header=None).to_numpy()).t().long()
        # random_network_edgelist_backward = torch.vstack((random_network_edgelist_forward[1,:], random_network_edgelist_forward[0,:]))

        max_interactions = self.agents_random_interactions.max() + 1
        agents_interactions_matrix = torch.ones(self.params['num_agents'], max_interactions).long()
        agents_interactions_matrix[torch.arange(self.params['num_agents']), self.agents_random_interactions] = 0
        agents_interactions_matrix = agents_interactions_matrix.cumprod(1)
        agents_edges = (agents_interactions_matrix * (torch.arange(self.params['num_agents']) + 1).view(-1, 1)).view(-1)  # +1 is for correcting for 0 index
        agents_edges = agents_edges[agents_edges > 0]
        agents_edges = agents_edges - 1  # -1 is for correcting for 0 index
        agents_edges_chosen = agents_edges[torch.randperm(agents_edges.shape[0]).long()]
        random_network_edgelist_forward = torch.vstack((agents_edges_chosen[1:], agents_edges_chosen[:-1]))
        random_network_edgelist_backward = torch.vstack(
            (random_network_edgelist_forward[1, :], random_network_edgelist_forward[0, :]))

        random_network_edgelist = torch.hstack((random_network_edgelist_forward, random_network_edgelist_backward))
        random_network_edgeattr_type = self.network_type_dict['random']*torch.ones(random_network_edgelist.shape[1]).long() #22 is the index for random network
        random_network_edgeattr_B_n = torch.ones(random_network_edgelist.shape[1]).float()*self.B_n['random']
        random_network_edgeattr = torch.vstack((random_network_edgeattr_type, random_network_edgeattr_B_n))

        all_edgelist = torch.hstack((self.household_network_edgelist, all_occupation_network_edgelist, random_network_edgelist))
        all_edgeattr = torch.hstack((self.household_network_edgeattr, all_occupation_network_edgeattr, random_network_edgeattr))

        agents_data = Data(x, edge_index=all_edgelist, edge_attr=all_edgeattr, t=t, agents_mean_interactions = self.agents_mean_interactions)
        lam_t = self.net(agents_data) #TODO
        prob_not_infected = torch.exp(-lam_t)
        p = torch.hstack((prob_not_infected, 1-prob_not_infected))
        potentially_infected_now = torch.distributions.Categorical(probs=p).sample()
        infected_now_pre_vaccine = torch.logical_and(potentially_infected_now, torch.logical_not(self.agents_infected_index[t]))

        infected_now = torch.logical_and(infected_now_pre_vaccine, torch.logical_not(self.agents_vaccination_effectiveness[t, :])).view(-1)

        if self.params['debug']:
            print('Infected now before vaccination: ', infected_now_pre_vaccine.sum())
            print('Infected now after vaccination: ', infected_now.sum())
            if infected_now_pre_vaccine.sum() > 0:
                print('percentage protection: ', (infected_now.sum()/infected_now_pre_vaccine.sum())*100.0)

        disease_probs_agents = self.disease_probs_age[self.agents_ages[infected_now.bool()].long(),:,:]
        # agents_stages_onehot = torch.zeros(agents_stages[t, infected_now.bool()].shape[0], num_stages)
        # agents_stages_onehot.scatter_(1, agents_stages[t, infected_now.bool()].view(-1,1).long(), 1)
        agents_stages_onehot = make_one_hot(self.agents_stages[t, infected_now.bool()], self.params['num_stages'])
        probs = torch.bmm(disease_probs_agents.permute(0, 2, 1).float(), agents_stages_onehot.float().unsqueeze(2)).squeeze() # torch.bmm(f.permute(0, 2, 1),e.unsqueeze(2)).squeeze()
        self.agents_stages[t+1,infected_now.bool()] = torch.distributions.Categorical(probs=probs).sample()
        self.agents_infected_index[t+1, infected_now.bool()] = True
        self.agents_infected_time[t+1, infected_now.bool()] = t

        #Add next stages and next times for these agents too and also outside the for loop for the first infected agents!
        if self.params['debug']:
            print('There are {} agents infected today'.format(infected_now.sum()))
        if infected_now.sum() > 0:
            self._set_next_stages_times(infected_now, self.agents_stages[t+1,:], t) 

        #**********************************************************************************
        #5. Disease dynamics
        #Get agents ready for transition
        transition_agents_indices = torch.logical_and(torch.logical_and(torch.logical_and(self.agents_next_stage_times == t, self.agents_infected_index[t]), self.agents_stages[t, :] != 8), 
        self.agents_stages[t, :] != 10) #No DEATH people and No RECOVERED people
        if transition_agents_indices.sum() > 0:
            #Get their tp1 stages
            self.agents_stages[t+1, transition_agents_indices.bool()] = self.agents_next_stages[transition_agents_indices.bool()]
            self._set_next_stages_times(transition_agents_indices, self.agents_stages[t+1,:], t)
        #**********************************************************************************
        #6. DEN dynamics
        self.den_contacts.append(torch.clone(all_edgelist))
        #**********************************************************************************
        self.current_time += 1
        
    def log_results(self):
        pass

    def _get_lam_gamma_integrals(self, mu, sigma, t):
        b = sigma * sigma / mu
        a = mu / b
        res = [(gamma.cdf(t_i, a=a, loc=0, scale=b) - gamma.cdf(t_i-1, a=a, loc=0, scale=b)) for t_i in range(t)]
        return torch.tensor(res).float()

    def _set_next_stages_times(self, agents_indices, stages_t, t):
        agents_stages_onehot = make_one_hot(stages_t[agents_indices.bool()], self.params['num_stages'])
        disease_probs_agents = self.disease_probs_age[self.agents_ages[agents_indices.bool()].long(),:,:]
        probs = torch.bmm(disease_probs_agents.permute(0, 2, 1).float(), agents_stages_onehot.float().unsqueeze(2)).squeeze()
        self.agents_next_stages[agents_indices.bool()] = torch.distributions.Categorical(probs=probs).sample()
        #Get next times (direct update without tp1 staging)
        dist_transition_times_agents_mat = self.disease_transition_times_dists_age[self.agents_ages[agents_indices.bool()].long(),:,:]
        mu_transition_times_agents_mat = self.disease_transition_times_mu_age[self.agents_ages[agents_indices.bool()].long(),:,:]
        sigma_transition_times_agents_mat = self.disease_transition_times_sigma_age[self.agents_ages[agents_indices.bool()].long(),:,:]
        
        dist_transition_times_agents = torch.bmm(dist_transition_times_agents_mat.permute(0, 2, 1).float(), agents_stages_onehot.float().unsqueeze(2)).squeeze()
        if len(dist_transition_times_agents.shape) < 2:
            dist_transition_times_agents = dist_transition_times_agents.view(1,-1)
        dist_transition_times_agents = dist_transition_times_agents.gather(1, self.agents_next_stages[agents_indices.bool()].long().view(-1, 1)).view(-1)

        mu_transition_times_agents = torch.bmm(mu_transition_times_agents_mat.permute(0, 2, 1).float(), agents_stages_onehot.float().unsqueeze(2)).squeeze()
        if len(mu_transition_times_agents.shape) < 2:
            mu_transition_times_agents = mu_transition_times_agents.view(1,-1)
        mu_transition_times_agents = mu_transition_times_agents.gather(1, self.agents_next_stages[agents_indices.bool()].long().view(-1, 1)).view(-1)

        sigma_transition_times_agents = torch.bmm(sigma_transition_times_agents_mat.permute(0, 2, 1).float(), agents_stages_onehot.float().unsqueeze(2)).squeeze()
        if len(sigma_transition_times_agents.shape) < 2:
            sigma_transition_times_agents = sigma_transition_times_agents.view(1,-1)
        sigma_transition_times_agents = sigma_transition_times_agents.gather(1, self.agents_next_stages[agents_indices.bool()].long().view(-1, 1)).view(-1)
        
        # agents_to_set = agents_indices.nonzero().view(-1) #agents_next_stage_times[agents_indices.bool()]
        agents_to_set = agents_indices.nonzero(as_tuple=True)[0]
        #Gamma agents #coded by 0
        gamma_indices = dist_transition_times_agents == 0
        gamma_mu = mu_transition_times_agents[gamma_indices.bool()]
        gamma_sigma = sigma_transition_times_agents[gamma_indices.bool()]
        gamma_a = (gamma_mu*gamma_mu)/(gamma_sigma*gamma_sigma)
        gamma_b = (gamma_sigma*gamma_sigma)/gamma_mu
        gamma_next_times = torch.distributions.Gamma(gamma_a, gamma_b).sample().round().long() #TODO Check this
        agents_to_set_gamma = agents_to_set[gamma_indices.bool()]
        self.agents_next_stage_times[agents_to_set_gamma.long()] = (t + 1+ gamma_next_times).long() #Got infected at end of day
        
        #Bernoulli agents #coded by 1
        bernoulli_indices = dist_transition_times_agents == 1
        bernoulli_mu = mu_transition_times_agents[bernoulli_indices.bool()]
        bernoulli_low = torch.floor(bernoulli_mu)
        bernoulli_high = torch.ceil(bernoulli_mu)
        bernoulli_p = (bernoulli_low - bernoulli_mu)/(bernoulli_low - bernoulli_high)
        bernoulli_next_times = bernoulli_low + torch.distributions.Bernoulli(probs=bernoulli_p).sample().long() #TODO Check this
        agents_to_set_bernoulli = agents_to_set[bernoulli_indices.bool()]    
        self.agents_next_stage_times[agents_to_set_bernoulli.long()] = (t + 1 + bernoulli_next_times).long() #Got infected at end of day

########################################################################################################
    def collect_results(self):
        plot_data = np.zeros((self.params['num_steps'],11))
        for t in range(self.params['num_steps']):
            plot_data[t,:] = make_one_hot(self.agents_stages[t, :], self.params['num_stages']).sum(axis=0).numpy()
        plot_df_columns = list(self.params['stages'])
        plot_df = pd.DataFrame(data = plot_data, columns=plot_df_columns)
        plot_df['INFECTED'] = plot_df.apply(lambda x: sum(x), axis=1)
        plot_df['INFECTED'] -= plot_df['SUSCEPTIBLE']
        plot_df['ACTIVE'] = plot_df['INFECTED'] - plot_df['RECOVERED'] - plot_df['DEATH']
        return plot_df
    

if __name__ == "__main__":
    import yaml
    import random
    device = torch.device("cpu")
    params = yaml.safe_load(open('Data/generated_params.yaml', 'r'))
    params['seed'] = 123
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed']) 
    abm = TorchABMCovid(params, device)
    for i in range(params['num_steps']):
        abm.step()
