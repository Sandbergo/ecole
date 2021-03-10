import gzip
import pickle
import numpy as np
import ecole
import torch
import torch.nn.functional as F
import torch_geometric
import os
from pathlib import Path
from utilities import log


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


class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite` 
    observation function in a format understood by the pytorch geometric data handlers.
    """
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features,
                 candidates, candidate_choice, candidate_scores):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = len(candidates)
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores

    def __inc__(self, key, value):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs 
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, sample_action, sample_action_set, sample_scores = sample

        constraint_features, (edge_indices, edge_features), variable_features = sample_observation
        constraint_features = torch.from_numpy(constraint_features.astype(np.float32))
        edge_indices = torch.from_numpy(edge_indices.astype(np.int64))
        edge_features = torch.from_numpy(edge_features.astype(np.float32)).view(-1, 1)
        variable_features = torch.from_numpy(variable_features.astype(np.float32))
        
        # We note on which variables we were allowed to branch, the scores as well as the choice 
        # taken by strong branching (relative to the candidates)
        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])
        candidate_choice = torch.where(candidates == sample_action)[0][0]

        graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features,
                                  candidates, candidate_choice, candidate_scores)

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0]+variable_features.shape[0]

        return graph


class GNNPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 19

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        
        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)
        return output
    

class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need 
    to provide the exact form of the messages being passed.
    """
    def __init__(self):
        super().__init__('add')
        emb_size = 64

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size)
        )
        self.post_conv_module = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size)
        )
        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]), 
                                node_features=(left_features, right_features), edge_features=edge_features)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(self.feature_module_left(node_features_i) 
                                           + self.feature_module_edge(edge_features) 
                                           + self.feature_module_right(node_features_j))
        return output


def process(policy, data_loader, optimizer=None):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_acc = 0

    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            batch = batch.to(DEVICE)
            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            logits = policy(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
            # Index the results by the candidates, and split and pad them
            logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            # Compute the usual cross-entropy classification loss
            loss = F.cross_entropy(logits, batch.candidate_choices)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
            true_bestscore = true_scores.max(dim=-1, keepdims=True).values
            
            predicted_bestindex = logits.max(dim=-1, keepdims=True).indices
            accuracy = (true_scores.gather(-1, predicted_bestindex) == true_bestscore).float().mean().item()

            mean_loss += loss.item() * batch.num_graphs
            mean_acc += accuracy * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    mean_acc /= n_samples_processed
    return mean_loss, mean_acc


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    """
    This utility function splits a tensor and pads each split to make them all the same size, then stacks them.
    """
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack([F.pad(slice_, (0, max_pad_size-slice_.size(0)), 'constant', pad_value)
                          for slice_ in output], dim=0)
    return output


if __name__ == "__main__":
    # -- MAKE DATASET -- #

    MAX_SAMPLES = 10_000

    Path('log/').mkdir(exist_ok=True)
    logfile = 'log/log.txt'

    instances = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05)

    # We can pass custom SCIP parameters easily
    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}

    # Note how we can tuple observation functions to return complex state information
    env = ecole.environment.Branching(observation_function=(ExploreThenStrongBranch(expert_probability=0.20), 
                                                            ecole.observation.NodeBipartite()), 
                                    scip_params=scip_parameters)
    # TODO: original expert_prob=0.05
    # This will seed the environment for reproducibility
    env.seed(0)

    episode_counter, sample_counter = 0, 0
    Path('samples/').mkdir(exist_ok=True)

    # We will solve problems (run episodes) until we have saved enough samples
    max_samples_reached = False
    while not max_samples_reached:
        episode_counter += 1

        observation, action_set, _, done, _ = env.reset(next(instances))
        while not done:
            (scores, scores_are_expert), node_observation = observation
            node_observation = (node_observation.row_features,
                                (node_observation.edge_features.indices, 
                                node_observation.edge_features.values),
                                node_observation.column_features)
        
            action = action_set[scores[action_set].argmax()]

            # Only save samples if they are coming from the expert (strong branching)
            if scores_are_expert and not max_samples_reached:
                sample_counter += 1
                data = [node_observation, action, action_set, scores]
                filename = f'samples/sample_{sample_counter}.pkl'

                with gzip.open(filename, 'wb') as f:
                    pickle.dump(data, f)

                # If we collected enough samples, we finish the current episode but stop saving samples
                if sample_counter >= MAX_SAMPLES:
                    max_samples_reached = True

            observation, action_set, _, done, _ = env.step(action)

        log(f"Episode {episode_counter}, {sample_counter} / {MAX SAMPLES} samples collected so far")


    # -- TRAIN -- #
    LEARNING_RATE = 0.001
    NB_EPOCHS = 50
    PATIENCE = 10
    EARLY_STOPPING = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sample_files = [str(path) for path in Path('samples/').glob('sample_*.pkl')]
    train_files = sample_files[:int(0.8*len(sample_files))]
    valid_files = sample_files[int(0.8*len(sample_files)):]

    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.data.DataLoader(train_data, batch_size=32, shuffle=True)
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=128, shuffle=False)

    policy = GNNPolicy().to(DEVICE)

    # observation = train_data[0].to(DEVICE)

    # logits = policy(observation.constraint_features, observation.edge_index, observation.edge_attr, observation.variable_features)
    # action_distribution = F.softmax(logits[observation.candidates], dim=-1)

    # print(action_distribution)

    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    for epoch in range(NB_EPOCHS):
        log(f"Epoch {epoch+1}")
        
        train_loss, train_acc = process(policy, train_loader, optimizer)
        log(f"Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}")

        valid_loss, valid_acc = process(policy, valid_loader, None)
        log(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}")

    torch.save(policy.state_dict(), 'trained_params.pkl')


    # -- EVALUATE -- #

    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}
    env = ecole.environment.Branching(observation_function=ecole.observation.NodeBipartite(), 
                                    information_function={"nb_nodes": ecole.reward.NNodes(), 
                                                            "time": ecole.reward.SolvingTime()}, 
                                    scip_params=scip_parameters)
    default_env = ecole.environment.Configuring(observation_function=None,
                                                information_function={"nb_nodes": ecole.reward.NNodes(), 
                                                                    "time": ecole.reward.SolvingTime()}, 
                                                scip_params=scip_parameters)

    instances = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05)
    for instance_count, instance in zip(range(20), instances):
        # Run the GNN brancher
        nb_nodes, time = 0, 0
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

    instances = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05)
    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}
    env = ecole.environment.Branching(observation_function=ecole.observation.NodeBipartite(), 
                                    information_function={"nb_nodes": ecole.reward.NNodes().cumsum(), 
                                                            "time": ecole.reward.SolvingTime().cumsum()}, 
                                    scip_params=scip_parameters)
    default_env = ecole.environment.Configuring(observation_function=None,
                                                information_function={"nb_nodes": ecole.reward.NNodes().cumsum(), 
                                                                    "time": ecole.reward.SolvingTime().cumsum()}, 
                                                scip_params=scip_parameters)

    for instance_count, instance in zip(range(20), instances):
        # Run the GNN brancher
        observation, action_set, _, done, info = env.reset(instance)
        while not done:
            with torch.no_grad():
                observation = (torch.from_numpy(observation.row_features.astype(np.float32)).to(DEVICE),
                            torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(DEVICE), 
                            torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(DEVICE),
                            torch.from_numpy(observation.column_features.astype(np.float32)).to(DEVICE))
                logits = policy(*observation)
                action = action_set[logits[action_set.astype(np.int64)].argmax()]
                observation, action_set, _, done, info = env.step(action)
        nb_nodes = info['nb_nodes']
        time = info['time']

        # Run SCIP's default brancher
        default_env.reset(instance)
        _, _, _, _, default_info = default_env.step({})

        log(f"Instance {instance_count: >3} | SCIP nb nodes    {int(default_info['nb_nodes']): >4d}  | SCIP time   {default_info['time']: >6.2f} ")
        log(f"             | GNN  nb nodes    {int(nb_nodes): >4d}  | GNN  time   {time: >6.2f} ")
        log(f"             | Gain         {100*(1-nb_nodes/default_info['nb_nodes']): >8.2f}% | Gain      {100*(1-time/default_info['time']): >8.2f}%")
