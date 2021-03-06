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



if __name__ == "__main__":
    Path('examples/log/').mkdir(exist_ok=True)
    log = Logger(filename='examples/log/test_log.txt')

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device {DEVICE}')

    policy = GNNPolicy().to(DEVICE)
    policy.load_state_dict(torch.load('examples/models/trained_params.pkl'))
    

    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}
    env = ecole.environment.Branching(observation_function=ecole.observation.NodeBipartite(), 
                                    information_function={"nb_nodes": ecole.reward.NNodes(), 
                                                            "time": ecole.reward.SolvingTime()}, 
                                    scip_params=scip_parameters)
    
    default_env = ecole.environment.Configuring(observation_function=None,
                                                information_function={"nb_nodes": ecole.reward.NNodes(), 
                                                                      "time": ecole.reward.SolvingTime()}, 
                                                scip_params=scip_parameters)
    
    generators = {
        'setcover':(
            ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05),
            ecole.instance.SetCoverGenerator(n_rows=1000, n_cols=1000, density=0.05),
            ecole.instance.SetCoverGenerator(n_rows=2000, n_cols=1000, density=0.05)
        ),
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
    for problem_type in generators.keys():
        i = 0
        for instances in generators[problem_type]:      
            log(f'\n{problem_type} size {i}\n')
            for instance_count, instance in zip(range(20), instances):
                
                # Run the GNN brancher
                nb_nodes, time = 0, 0
                default_env.reset(instance)
                observation, action_set, _, done, info = env.reset(instance)
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
                    time += info['time']

                # Run SCIP's default brancher
                default_env.reset(instance)
                _, _, _, _, default_info = default_env.step({})
                
                log(f"Instance {instance_count: >3} | SCIP nb nodes    {int(default_info['nb_nodes']): >4d}  | SCIP time   {default_info['time']: >6.2f} ")
                log(f"             | GNN  nb nodes    {int(nb_nodes): >4d}  | GNN  time   {time: >6.2f} ")
                log(f"             | Gain         {100*(1-nb_nodes/default_info['nb_nodes']): >8.2f}% | Gain      {100*(1-time/default_info['time']): >8.2f}%")
            i += 1