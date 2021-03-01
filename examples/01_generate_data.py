import ecole
import os
import gzip
import pickle
import numpy as np
from pathlib import Path
from utilities import Logger


class ExploreThenStrongBranch:
    """
    This custom observation function class will randomly return either strong branching scores (expensive expert) 
    or pseudocost scores (weak expert for exploration) when called at every node.
    """
    def __init__(self, expert_probability):
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

    # MAX_SAMPLES = 10_000

    Path('examples/log/').mkdir(exist_ok=True)
    log = Logger(filename='examples/log/data_log.txt')

    # We can pass custom SCIP parameters easily
    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}

    # Note how we can tuple observation functions to return complex state information
    env = ecole.environment.Branching(observation_function=(ExploreThenStrongBranch(expert_probability=0.2),
                                                            ecole.observation.NodeBipartite(),
                                                            ecole.observation.Khalil2016()), 
                                    scip_params=scip_parameters)
    # TODO: original expert_prob=0.05
    # This will seed the environment for reproducibility
    env.seed(0)

    PROBLEMS = {'setcover': (500, 1000, 0.05), 
                'cauctions': (100, 500, 0),
                'indset': (750, 4, 0),
                'faciliites': (100, 100, 5) }

    MAX_SAMPLES = 50_000  # 150000
    # VALID_SIZE = 10_000  # 30000
    # TEST_SIZE  = 10_000  # 30000
    TIME_LIMIT = 3_600  # 3600
    # node_limit = 500

    node_record_prob = 1.0

    basedir = "examples/data/samples/"

    generators = {
        #'setcover':ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05),
        'cauctions': ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=500),
        'indset': ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=100),
        'faciliites': ecole.instance.IndependentSetGenerator(n_nodes=500, graph_type="erdos_renyi"),
        }
    
    try:
        for problem_type in generators.keys():
            log(f'Generating {MAX_SAMPLES} {problem_type} instances')
            instances = generators[problem_type]
            episode_counter, sample_counter = 0, 0
            path = f'{basedir}/{problem_type}/train/'
            os.makedirs(Path(path), exist_ok=True)

            # We will solve problems (run episodes) until we have saved enough samples
            max_samples_reached = False
            while not max_samples_reached:
                episode_counter += 1

                observation, action_set, _, done, _ = env.reset(next(instances))
                while not done:
                    (scores, scores_are_expert), node_observation, khalil2016 = observation
                    print(len(node_observation.row_features), len(node_observation.row_features[0]))
                    print(len(node_observation.edge_features.indices), len(node_observation.edge_features.indices[0]))
                    node_observation = (node_observation.row_features,
                                        (node_observation.edge_features.indices, 
                                        node_observation.edge_features.values),
                                        node_observation.column_features)
                
                    action = action_set[scores[action_set].argmax()]
                    
                    print(len(khalil2016), len(khalil2016[0]) )
                    exit(0)
                    # Only save samples if they are coming from the expert (strong branching)
                    if scores_are_expert and not max_samples_reached:
                        sample_counter += 1
                        data = [node_observation, action, action_set, scores]
                        filename = f'{path}/sample_{sample_counter}.pkl'

                        with gzip.open(filename, 'wb') as f:
                            pickle.dump(data, f)
                        
                        # If we collected enough samples, we finish the current episode but stop saving samples
                        if sample_counter >= MAX_SAMPLES:
                            max_samples_reached = True

                    observation, action_set, _, done, _ = env.step(action)

                log(f"Episode {episode_counter}, {sample_counter} / {MAX_SAMPLES} samples collected so far")
    except Exception as e:
        log(repr(e))
        raise e
