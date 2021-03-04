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

# TODO: what do 

class MLPDataset(torch.utils.data.Dataset):
    def __init__(self, sample_files, weighing_scheme="sigmoidal_decay"):
        self.sample_files = sample_files
        self.weighing_scheme = weighing_scheme if weighing_scheme != "" else "constant"

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        is_root = "root" in self.sample_files[index]

        obss, target, obss_feats, _ = sample['obss']
        v, _, _ = obss
        sample_cand_scores = obss_feats['scores']
        sample_cands = np.where(sample_cand_scores != -1)[0]

        v_feats = v[sample_cands]
        v_feats = utilities._preprocess(v_feats, mode='min-max-2')

        cand_scores = sample_cand_scores[sample_cands]
        sample_action = np.where(sample_cands == target)[0][0]

        weight = obss_feats['depth']/sample['max_depth'] if sample['max_depth'] else 1.0
        if self.weighing_scheme == "sigmoidal_decay":
            weight = (1 + np.exp(-0.5))/(1 + np.exp(weight - 0.5))
        elif self.weighing_scheme == "constant":
            weight = 1.0
        else:
            raise ValueError(f"Unknown value for node weights: {self.weighing_scheme}")

        return v_feats, sample_action, cand_scores, weight


def load_batch(sample_batch):
    cand_featuress, sample_actions, cand_scoress, weights = list(zip(*sample_batch))

    n_cands = [cds.shape[0] for cds in cand_featuress]

    # convert to numpy arrays
    cand_featuress = np.concatenate(cand_featuress, axis=0)
    cand_scoress = np.concatenate(cand_scoress, axis=0)
    n_cands = np.array(n_cands)
    best_actions = np.array(sample_actions)
    weights = np.array(weights)

    # convert to tensors
    cand_featuress = torch.as_tensor(cand_featuress, dtype=torch.float32)
    cand_scoress = torch.as_tensor(cand_scoress, dtype=torch.float32)
    n_cands = torch.as_tensor(n_cands, dtype=torch.int32)
    best_actions = torch.as_tensor(sample_actions, dtype=torch.long)
    weights = torch.as_tensor(weights, dtype=torch.float32)

    return cand_featuress, n_cands, best_actions, cand_scoress, weights


class Model(torch.nn.Module):
    def initialize_parameters(self):
        for l in self.modules():
            if isinstance(l, torch.nn.Linear):
                torch.nn.init.orthogonal_(l.weight.data, gain=1)
                if l.bias is not None:
                    torch.nn.init.constant_(l.bias.data, 0)

    def save_state(self, filepath):
        torch.save(self.state_dict(), filepath)

    def restore_state(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))


class MLPPolicy(Model):
    def __init__(self):
        super(Model, self).__init__()

        self.n_input_feats = 92
        self.ff_size = 256

        self.activation = torch.nn.ReLU()
        # Original, seed 0
        self.output_module = nn.Sequential(
            nn.Linear(self.n_input_feats, self.ff_size, bias=True),
            self.activation,
            nn.Linear(self.ff_size, self.ff_size, bias=True),
            self.activation,
            nn.Linear(self.ff_size, self.ff_size, bias=True),
            self.activation,
            nn.Linear(self.ff_size, 1, bias=False),
        )
        """
        # seed 46
        self.output_module = nn.Sequential(
            nn.Linear(self.n_input_feats, self.ff_size*4, bias=True),
            self.activation,
            nn.Linear(self.ff_size*4, self.ff_size*2, bias=True),
            self.activation,
            nn.Linear(self.ff_size*2, self.ff_size, bias=True),
            self.activation,
            nn.Linear(self.ff_size, 1, bias=True),
        )
        # seed 35
        self.output_module = nn.Sequential(
            nn.Linear(self.n_input_feats, 512, bias=True),
            self.activation,
            nn.Linear(512, 256, bias=True),
            self.activation,
            nn.Linear(256, 64, bias=True),
            self.activation,
            nn.Linear(64, 1, bias=True),
        )
        
        
        
        # seed 70
        self.output_module = nn.Sequential(
            nn.Linear(self.n_input_feats, self.ff_size*4, bias=True),
            self.activation,
            nn.Linear(self.ff_size*4, self.ff_size*4, bias=True),
            self.activation,
            nn.Linear(self.ff_size*4, self.ff_size*4, bias=True),
            self.activation,
            nn.Linear(self.ff_size*4, self.ff_size*4, bias=True),
            self.activation,
            nn.Linear(self.ff_size*4, self.ff_size*4, bias=True),
            self.activation,
            nn.Linear(self.ff_size*4, self.ff_size*4, bias=True),
            self.activation,
            nn.Linear(self.ff_size*4, 1, bias=True),
        )"""

        # Seed 61
        self.output_module = nn.Sequential(
            nn.Linear(self.n_input_feats, 1, bias=True),
        )
        
        print(self.output_module)
        self.initialize_parameters()

    @staticmethod
    def pad_output(output, n_vars_per_sample, pad_value=-1e8):
        n_vars_max = torch.max(n_vars_per_sample)

        output = torch.split(
            tensor=output,
            split_size_or_sections=n_vars_per_sample.tolist(),
            dim=1,
        )

        output = torch.cat([
            F.pad(x,
                  pad=[0, n_vars_max - x.shape[1], 0, 0],
                  mode='constant',
                  value=pad_value)
            for x in output
        ], dim=0)

        return output

    def forward(self, inputs):
        features = inputs
        output = self.output_module(features)
        output = torch.reshape(output, [1, -1])
        # For benchmarking random, in log 35
        # output = torch.rand(output.shape, requires_grad=True)
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
            logits = policy(batch.khalil_features)
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

    LEARNING_RATE = 0.001
    NB_EPOCHS = 50
    PATIENCE = 10
    EARLY_STOPPING = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Path('examples/log/').mkdir(exist_ok=True)
    log = Logger(filename='examples/log/train_log.txt')

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

    torch.save(policy.state_dict(), 'examples/models/gnn_trained_params.pkl')


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
