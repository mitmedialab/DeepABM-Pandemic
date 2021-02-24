import torch

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

mean_int_ran_mu = torch.tensor(mean_int_occ_df['mu'].values).float().to(self.device)
mean_int_ran_sigma = torch.tensor(mean_int_occ_df['sigma'].values).float().to(self.device)

self.agents_random_interactions = torch.tensor([
    get_num_random_interactions(age, mean_int_ran_mu, mean_int_ran_sigma, self.params['child_upper_ix'], self.params['adult_upper_ix'])
    for age in self.agents_ages
])
######################################## in _init_ ########################

#If vaccine = self.agents_interactions[] has to be scaled appropriately

#In step:
max_interactions = self.agents_random_interactions.max() + 1
agents_interactions_matrix = torch.ones(self.params['num_agents'], max_interactions).long()
agents_interactions_matrix[torch.arange(self.params['num_agents']), self.agents_interactions] = 0
agents_interactions_matrix = agents_interactions_matrix.cumprod(1)
agents_edges = (agents_interactions_matrix * (torch.arange(self.params['num_agents'])+1).view(-1,1)).view(-1) #+1 is for correcting for 0 index
agents_edges = agents_edges[agents_edges > 0]
agents_edges = agents_edges - 1  #-1 is for correcting for 0 index
agents_edges_chosen = agents_edges[torch.randperm(agents_edges.shape[0]).long()]
random_network_edgelist_forward = torch.vstack((agents_edges_chosen[1:], agents_edges_chosen[:-1]))
random_network_edgelist_backward = torch.vstack((random_network_edgelist_forward[1,:], random_network_edgelist_forward[0,:]))