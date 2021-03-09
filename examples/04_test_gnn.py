import gzip
import pickle
import numpy as np
import ecole
import torch
import torch.nn.functional as F
import torch_geometric
import os
from pathlib import Path
from utilities import Logger

from models.gnn import process, GNNPolicy



class ExploreThenStrongBranch:
    """
    This custom observation function class will randomly return either strong branching scores (expensive expert) 
    or pseudocost scores (weak expert for exploration) when called at every node.
    """
    def __init__(self, expert_probability=0.05):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()
        
    def before_reset(self, model):
        """
        This function will be called at initialization of the environment (before dynamics are reset).
        """
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        """
        Should we return strong branching or pseudocost scores at time node?
        """
        probabilities = [1-self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model, done), True)
        else:
            return (self.pseudocosts_function.extract(model, done), False)



if __name__ == "__main__":
    Path('examples/log/').mkdir(exist_ok=True)
    log = Logger(filename='examples/log/test_log.txt')

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device: {DEVICE}')

    policy = GNNPolicy().to(DEVICE)
    policy.load_state_dict(torch.load('examples/models/gnn_trained_params_cauctions.pkl'))
    

    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 100} # TODO: revert to 2700
    env = ecole.environment.Branching(observation_function=ecole.observation.NodeBipartite(), 
                                    information_function={"nb_nodes": ecole.reward.NNodes(), 
                                                            "time": ecole.reward.SolvingTime()}, 
                                    scip_params=scip_parameters)
    
    default_env = ecole.environment.Branching(observation_function=ExploreThenStrongBranch(expert_probability=0.05),
                                                information_function={"nb_nodes": ecole.reward.NNodes(), 
                                                                      "time": ecole.reward.SolvingTime()}, 
                                                scip_params=scip_parameters)
    
    generators = {
        #'setcover':(
        #    ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05),
        #    ecole.instance.SetCoverGenerator(n_rows=1000, n_cols=1000, density=0.05),
        #    ecole.instance.SetCoverGenerator(n_rows=2000, n_cols=1000, density=0.05)
        #),
        'cauctions': (
            ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=500),
            ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=1000),
            ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=1500)
        ),
        
        'indset': (
            ecole.instance.IndependentSetGenerator(n_nodes=750, graph_type="erdos_renyi"),
            ecole.instance.IndependentSetGenerator(n_nodes=1000, graph_type="erdos_renyi"),
            ecole.instance.IndependentSetGenerator(n_nodes=1500, graph_type="erdos_renyi"),
            
        ),
        'facilitites': (
            ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=100),
            ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=200),
            ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=400),
        )
        }
    sizes = ['small', 'medium', 'large']
    for problem_type in generators.keys():
        i = 0
        scip_time,gnn_time = [],[]
        scip_nodes,gnn_nodes = [],[]
        for instances in generators[problem_type]:      
            log(f'------ {problem_type}, {sizes[i]} ------')
            for instance_count, instance in zip(range(5), instances):
                
                # Run the GNN brancher
                nb_nodes, time = 0, 0
                #default_env.reset(instance)
                """observation, action_set, _, done, info = env.reset(instance)
                
                nb_nodes += info['nb_nodes']
                time += info['time']
                while not done:
                    with torch.no_grad():
                        observation = (torch.from_numpy(observation.row_features.astype(np.float32)).to(DEVICE),
                                    torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(DEVICE), 
                                    torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(DEVICE),
                                    torch.from_numpy(observation.column_features.astype(np.float32)).to(DEVICE))
                        logits = policy(*observation)
                        action = action_set[logits[action_set.astype(np.int64)].argmax()]
                        observation, action_set, _, done, info = env.step(action)
                    nb_nodes += info['nb_nodes']
                    time += info['time']"""
              
                # Run SCIP's default brancher
                default_nb_nodes = 0
                default_time = 0
                
                # default_env.reset(instance)
                #_, _, _, done, default_info = default_env.step({})
                #print(done)
                observation, action_set, _, done, info = default_env.reset(instance)
                while not done:
                    (scores) = observation
                    print(type(scores))
                    action = action_set[observation[action_set].argmax()]

                    observation, action_set, _, done, info = default_env.step(action)
                    default_nb_nodes += info['nb_nodes']
                    default_time += info['time']
                
                #log(f"Instance {instance_count: >3} | SCIP nb nodes    {int(default_info['nb_nodes']): >4d}  | SCIP time   {default_info['time']: >6.2f} ")
                #log(f"             | GNN  nb nodes    {int(nb_nodes): >4d}  | GNN  time   {time: >6.2f} ")
                #log(f"             | Gain         {100*(1-nb_nodes/default_info['nb_nodes']): >8.2f}% | Gain      {100*(1-time/default_info['time']): >8.2f}%")
                scip_nodes.append(default_nb_nodes)
                scip_time.append(default_time)
                gnn_nodes.append(nb_nodes)
                gnn_time.append(time)
            log(f"SCIP nb nodes    {int(np.mean(scip_nodes)): >4d}  | SCIP time   {np.mean(scip_nodes): >6.2f} ")
            log(f"GNN nb nodes     {int(np.mean(gnn_nodes)): >4d}  |  GNN time   {np.mean(gnn_nodes): >6.2f} ")
            #log(f"             | completed...")
                
            i += 1
