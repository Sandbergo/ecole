import ecole

if __name__ == "__main__":
    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}

    default_env = ecole.environment.Branching(observation_function=ecole.observation.Pseudocosts(),
                                                information_function={"nb_nodes": ecole.reward.NNodes().cumsum(), 
                                                                      "time": ecole.reward.SolvingTime().cumsum()}, 
                                                scip_params=scip_parameters)

    generators = {
        'setcover': ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05),
        'cauctions': ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=500),
        'indset': ecole.instance.IndependentSetGenerator(n_nodes=500, graph_type="erdos_renyi", affinity=4),
        'facilities': ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=100)
        }
    
    for problem_type, generator in generators.items():
        print(f'    Problem: {problem_type}')
        for instance_count, instance in zip(range(3), generator):
        
            observation, action_set, _, done, info = default_env.reset(instance)
            
            while not done:
                scores = observation
                action = action_set[observation[action_set].argmax()]
                observation, action_set, _, done, info = default_env.step(action)

            print(f"SCIP nb nodes  {int(info['nb_nodes']): >4d}  | SCIP time {info['time']: >6.2f} ")
