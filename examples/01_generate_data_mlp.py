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


class MLPObservation(ecole.observation.Khalil2016):
    """
    This custom observation function class implements Gupta et al. (2020) MLP
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
        obs = super().extract(model, done)
        # Apply scaling
        column_max_abs = np.abs(obs.column_features).max(0)
        obs.column_features[:] /= column_max_abs
        row_max_abs = np.abs(obs.row_features).max(0)
        obs.row_features[:] /= row_max_abs
        # Return the updated observation
        return obs

"""
File adapted from https://github.com/ds4dm/learn2branch
"""
import os
import argparse
import pickle
import glob
import shutil
import gzip
import math
import numpy as np
import multiprocessing as mp

import pyscipopt as scip
import utilities


class VanillaFullstrongBranchingDataCollector(scip.Branchrule):
    """
    Implements branching policy to be used by SCIP such that data collection required for hybrid models is embedded in it.
    """
    def __init__(self, rng, query_expert_prob=0.60):
        self.khalil_root_buffer = {}
        self.obss = []
        self.targets = []
        self.obss_feats = []
        self.exploration_policy = "pscost"
        self.query_expert_prob = query_expert_prob
        self.rng = rng
        self.iteration_counter = 0

    def branchinit(self):
        self.ndomchgs = 0
        self.ncutoffs = 0
        self.khalil_root_buffer = {}

    def branchexeclp(self, allowaddcons):
        self.iteration_counter += 1

        query_expert = self.rng.rand() < self.query_expert_prob
        if query_expert or self.model.getNNodes() == 1:
            candidate_vars, *_ = self.model.getPseudoBranchCands()
            candidate_mask = [var.getCol().getLPPos() for var in candidate_vars]

            state = utilities.extract_state(self.model)
            state_khalil = utilities.extract_khalil_variable_features(self.model, candidate_vars, self.khalil_root_buffer)

            result = self.model.executeBranchRule('vanillafullstrong', allowaddcons)
            cands_, scores, npriocands, bestcand = self.model.getVanillafullstrongData()
            best_var = cands_[bestcand]

            self.add_obs(best_var, (state, state_khalil), (cands_, scores))
            if self.model.getNNodes() == 1:
                self.state = [state, state_khalil, self.obss[0]]

            self.model.branchVar(best_var)
            result = scip.SCIP_RESULT.BRANCHED
        else:
            result = self.model.executeBranchRule(self.exploration_policy, allowaddcons)

        # fair node counting
        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1

        return {'result': result}

    def add_obs(self, best_var, state_, cands_scores=None): # TODO: return to normalcy
        """
        Adds sample to the `self.obs` to be processed later at the end of optimization.
        Parameters
        ----------
            best_var : pyscipopt.Variable
                object representing variable in LP
            state_ : tuple
                extracted features of constraints and variables at a node
            cands_scores : np.array
                scores of each of the candidate variable on which expert policy was executed
        Return
        ------
        (bool): True if sample is added succesfully. False o.w.
        """
        if self.model.getNNodes() == 1:
            self.obss = []
            self.targets = []
            self.obss_feats = []
            self.map = sorted([x.getCol().getIndex() for x in self.model.getVars(transformed=True)])

        cands, scores = cands_scores
        # Do not record inconsistent scores. May happen if SCIP was early stopped (time limit).
        if any([s < 0 for s in scores]):
            return False

        state, state_khalil = state_
        var_features = state[2]['values']
        cons_features = state[0]['values']
        edge_features = state[1]

        # add more features to variables
        cands_index = [x.getCol().getIndex() for x in cands]
        khalil_features = -np.ones((var_features.shape[0], state_khalil.shape[1]))
        cand_ind = np.zeros((var_features.shape[0], 1))
        khalil_features[cands_index] = state_khalil
        cand_ind[cands_index] = 1
        var_features = np.concatenate([var_features, khalil_features, cand_ind], axis=1)

        tmp_scores = -np.ones(len(self.map))
        if scores:
            tmp_scores[cands_index] = scores

        self.targets.append(best_var.getCol().getIndex())
        self.obss.append([var_features, cons_features, edge_features])
        print(self.obss.shape)
        depth = self.model.getCurrentNode().getDepth()
        self.obss_feats.append({'depth': depth, 'scores': np.array(tmp_scores), 'iteration': self.iteration_counter})

        return True


def make_samples(in_queue, out_queue):
    """
    Worker loop: fetch an instance, run an episode and record samples.
    Parameters
    ----------
    in_queue : multiprocessing.Queue
        Input queue from which orders are received.
    out_queue : multiprocessing.Queue
        Output queue in which to send samples.
    """
    while True:
        episode, instance, seed, time_limit, outdir, rng = in_queue.get()

        m = scip.Model()
        m.setIntParam('display/verblevel', 0)
        m.readProblem(f'{instance}')
        utilities.init_scip_params(m, seed=seed)
        m.setIntParam('timing/clocktype', 2)
        m.setRealParam('limits/time', time_limit)
        m.setLongintParam('limits/nodes', node_limit)

        branchrule = VanillaFullstrongBranchingDataCollector(rng, node_record_prob)
        m.includeBranchrule(
            branchrule=branchrule,
            name="Sampling branching rule", desc="",
            priority=666666, maxdepth=-1, maxbounddist=1)

        m.setBoolParam('branching/vanillafullstrong/integralcands', True)
        m.setBoolParam('branching/vanillafullstrong/scoreall', True)
        m.setBoolParam('branching/vanillafullstrong/collectscores', True)
        m.setBoolParam('branching/vanillafullstrong/donotbranch', True)
        m.setBoolParam('branching/vanillafullstrong/idempotent', True)

        out_queue.put({
            "type": 'start',
            "episode": episode,
            "instance": instance,
            "seed": seed
        })

        m.optimize()
        # data storage - root and node data are saved separately.
        # node data carries a reference to the root filename.
        if m.getNNodes() >= 1 and len(branchrule.obss) > 0 :
            filenames = []
            max_depth = max(x['depth'] for x in branchrule.obss_feats)
            stats = {'nnodes': m.getNNodes(), 'time': m.getSolvingTime(), 'gap': m.getGap(), 'nobs': len(branchrule.obss)}

            # prepare root data
            sample_state, sample_khalil_state, root_obss = branchrule.state
            sample_cand_scores = branchrule.obss_feats[0]['scores']
            sample_cands = np.where(sample_cand_scores != -1)[0]
            sample_cand_scores = sample_cand_scores[sample_cands]
            cand_choice = np.where(sample_cands == branchrule.targets[0])[0][0]

            root_filename = f"{outdir}/sample_root_0_{episode}.pkl"

            filenames.append(root_filename)
            with gzip.open(root_filename, 'wb') as f:
                pickle.dump({
                    'type': 'root',
                    'episode': episode,
                    'instance': instance,
                    'seed': seed,
                    'stats': stats,
                    'root_state': [sample_state, sample_khalil_state, sample_cands, cand_choice, sample_cand_scores],
                    'obss': [branchrule.obss[0], branchrule.targets[0], branchrule.obss_feats[0], None],
                    'max_depth': max_depth
                    }, f)

            # node data
            for i in range(1, len(branchrule.obss)):
                iteration_counter = branchrule.obss_feats[i]['iteration']
                filenames.append(f"{outdir}/sample_node_{iteration_counter}_{episode}.pkl")
                with gzip.open(filenames[-1], 'wb') as f:
                    pickle.dump({
                        'type' : 'node',
                        'episode': episode,
                        'instance': instance,
                        'seed': seed,
                        'stats': stats,
                        'root_state': f"{outdir}/sample_root_0_{episode}.pkl",
                        'obss': [branchrule.obss[i], branchrule.targets[i], branchrule.obss_feats[i], None],
                        'max_depth': max_depth
                    }, f)

            out_queue.put({
                "type": "done",
                "episode": episode,
                "instance": instance,
                "seed": seed,
                "filenames": filenames,
                "nnodes": len(filenames),
            })

        m.freeProb()


def send_orders(orders_queue, instances, seed, time_limit, outdir, start_episode):
    """
    Worker loop: fetch an instance, run an episode and record samples.
    Parameters
    ----------
    orders_queue : multiprocessing.Queue
        Input queue from which orders are received.
    instances : list
        list of filepaths of instances which are solved by SCIP to collect data
    seed : int
        initial seed to insitalize random number generator with
    time_limit : int
        maximum time for which to solve an instance while collecting data
    outdir : str
        directory where to save data
    start_episode : int
        episode to resume data collection. It is used if the data collection process was stopped earlier for some reason.
    """
    rng = np.random.RandomState(seed)
    episode = 0
    while True:
        instance = rng.choice(instances)
        seed = rng.randint(2**32)
        # already processed; for a broken process; for root dataset to not repeat instances and seed
        if episode <= start_episode:
            episode += 1
            continue

        orders_queue.put([episode, instance, seed, time_limit, outdir, rng])
        episode += 1


def collect_samples(instances, outdir, rng, n_samples, n_jobs, time_limit):
    """
    Worker loop: fetch an instance, run an episode and record samples.
    Parameters
    ----------
    instances : list
        filepaths of instances which will be solved to collect data
    outdir : str
        directory where to save data
    rng : np.random.RandomState
        random number generator
    n_samples : int
        total number of samples to collect.
    n_jobs : int
        number of CPUs to utilize or number of instances to solve in parallel.
    time_limit : int
        maximum time for which to solve an instance while collecting data
    """
    os.makedirs(outdir, exist_ok=True)

    # start workers
    orders_queue = mp.Queue(maxsize=2*n_jobs)
    answers_queue = mp.SimpleQueue()
    workers = []
    for i in range(n_jobs):
        p = mp.Process(
                target=make_samples,
                args=(orders_queue, answers_queue),
                daemon=True)
        workers.append(p)
        p.start()

    # dir to keep samples temporarily; helps keep a prefect count
    tmp_samples_dir = f'{outdir}/tmp'
    os.makedirs(tmp_samples_dir, exist_ok=True)

    # if the process breaks due to some reason, resume from this last_episode.
    existing_samples = glob.glob(f"{outdir}/*.pkl")
    last_episode, last_i = -1, 0
    if existing_samples:
        last_episode = max(int(x.split("/")[-1].split(".pkl")[0].split("_")[-2]) for x in existing_samples) # episode is 2nd last
        last_i = max(int(x.split("/")[-1].split(".pkl")[0].split("_")[-1]) for x in existing_samples) # sample number is the last

    # start dispatcher
    dispatcher = mp.Process(
            target=send_orders,
            args=(orders_queue, instances, rng.randint(2**32), time_limit, tmp_samples_dir, last_episode),
            daemon=True)
    dispatcher.start()

    i = last_i # for a broken process
    in_buffer = 0
    while i <= n_samples:
        sample = answers_queue.get()

        if sample['type'] == 'start':
            in_buffer += 1

        if sample['type'] == 'done':
            for filename in sample['filenames']:
                x = filename.split('/')[-1].split(".pkl")[0]
                os.rename(filename, f"{outdir}/{x}.pkl")
                i += 1
                print(f"[m {os.getpid()}] {i} / {n_samples} samples written, ep {sample['episode']} ({in_buffer} in buffer).")

                if i == n_samples:
                    # early stop dispatcher (hard)
                    if dispatcher.is_alive():
                        dispatcher.terminate()
                        print(f"[m {os.getpid()}] dispatcher stopped...")
                    break

        if not dispatcher.is_alive():
            break

    # stop all workers (hard)
    for p in workers:
        p.terminate()

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # problem parameters
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )

    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=1,
    )
    args = parser.parse_args()

    train_size = 150000
    valid_size = 30000
    test_size = 30000
    time_limit = 3600
    node_limit = 500

    node_record_prob = 1.0

    basedir = "data/samples"
    # get instance filenames
    if args.problem == 'setcover':
        instances_train = glob.glob('data/instances/setcover/train_500r_1000c_0.05d/*.lp')
        instances_valid = glob.glob('data/instances/setcover/valid_500r_1000c_0.05d/*.lp')
        instances_test = glob.glob('data/instances/setcover/test_500r_1000c_0.05d/*.lp')
        out_dir = f'{basedir}/setcover/500r_1000c_0.05d'

    elif args.problem == 'cauctions':
        instances_train = glob.glob('data/instances/cauctions/train_100_500/*.lp')
        instances_valid = glob.glob('data/instances/cauctions/valid_100_500/*.lp')
        instances_test = glob.glob('data/instances/cauctions/test_100_500/*.lp')
        out_dir = f'{basedir}/cauctions/test' # TODO: f'{basedir}/cauctions/100_500'

    elif args.problem == 'indset':
        instances_train = glob.glob('data/instances/indset/train_750_4/*.lp')
        instances_valid = glob.glob('data/instances/indset/valid_750_4/*.lp')
        instances_test = glob.glob('data/instances/indset/test_750_4/*.lp')
        out_dir = f'{basedir}/indset/750_4'

    elif args.problem == 'facilities':
        instances_train = glob.glob('data/instances/facilities/train_100_100_5/*.lp')
        instances_valid = glob.glob('data/instances/facilities/valid_100_100_5/*.lp')
        instances_test = glob.glob('data/instances/facilities/test_100_100_5/*.lp')
        out_dir = f'{basedir}/facilities/100_100_5'
        time_limit = 600

    else:
        raise NotImplementedError

    print(f"{len(instances_train)} train instances for {train_size} samples")
    print(f"{len(instances_valid)} validation instances for {valid_size} samples")
    print(f"{len(instances_test)} test instances for {test_size} samples")

    rng = np.random.RandomState(args.seed + 1)
    collect_samples(instances_train, out_dir + "/train", rng, train_size, args.njobs, time_limit)
    print("Success: Train data collection")

    rng = np.random.RandomState(args.seed + 1)
    collect_samples(instances_valid, out_dir + "/valid", rng, valid_size, args.njobs, time_limit)
    print("Success: Valid data collection")

    rng = np.random.RandomState(args.seed + 1)
    collect_samples(instances_test, out_dir + "/test", rng, test_size, args.njobs, time_limit)
    print("Success: Test data collection")

    if args.problem == "indset":
        mediumvalid_size = 2000
        instances_mediumvalid = glob.glob('data/instances/indset/mediumvalid_1000_4/*.lp')
        out_dir = f'{basedir}/indset/1000_4'

        print(f"{len(instances_mediumvalid)} medium validation instances for {mediumvalid_size} samples")

        rng = np.random.RandomState(args.seed + 1)
        collect_samples(instances_mediumvalid, out_dir + "/mediumvalid", rng, mediumvalid_size, args.njobs, time_limit)
        print("Success: Medium validation data collection")

"""        
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
        #'indset': ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=100),
        #'faciliites': ecole.instance.IndependentSetGenerator(n_nodes=500, graph_type="erdos_renyi"),
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
                    # print(len(action_set))
                    #print('row: ', len(node_observation.row_features), len(node_observation.row_features[0]))
                    #print('edge ix: ', len(node_observation.edge_features.indices), len(node_observation.edge_features.indices[0]))
                    #print('edge val: ', len(node_observation.edge_features.values))
                    #print('col: ', len(node_observation.column_features), len(node_observation.column_features[0]))
                    
                    node_observation = (node_observation.row_features,
                                        (node_observation.edge_features.indices, 
                                        node_observation.edge_features.values),
                                        node_observation.column_features, 
                                        khalil2016)
                    # save khalil as well :) 
                    action = action_set[scores[action_set].argmax()]
                    np_kh = np.array([np.array(xi) for xi in khalil2016])
                    #print(np_kh.shape)
                    #print('Khalil: ', len(khalil2016), len(khalil2016[0]) )
                    #print(khalil2016[0], '\n\n')
                    #print(khalil2016[-1])
                    #print(action_set)

                    # exit(0)
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
                #exit(0)
    except Exception as e:
        log(repr(e))
        raise e
"""