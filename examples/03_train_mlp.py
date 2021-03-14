import gzip
import pickle
import numpy as np
import ecole
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import os
from pathlib import Path
from utilities import Logger
#from models.mlp import MLPDataset
#from models.mlp import process
#from models.mlp import MLPPolicy

# TODO: what do 


class MLPDataset(torch.utils.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """
    def __init__(self, sample_files):
        self.sample_files = sample_files

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, sample_action, sample_action_set, sample_scores = sample
        

        variable_features = torch.from_numpy(sample_observation.astype(np.float32))
        
        # We note on which variables we were allowed to branch, the scores as well as the choice 
        # taken by strong branching (relative to the candidates)
        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])
        candidate_choice = torch.where(candidates == sample_action)[0][0]
        sample_observation = torch.LongTensor(np.array(sample_observation, dtype=np.float32))
        # test
        
        print(sample_observation.shape, candidates.shape)
        exit(0)
        # \test

        v_feats = sample_observation[candidates]
        #v_feats = utilities._preprocess(v_feats, mode='min-max-2')

        cand_scores = sample_cand_scores[candidates]
        sample_action = np.where(candidates == target)[0][0]

        return v_feats, sample_action, cand_scores


def load_batch(sample_batch):
    cand_featuress, sample_actions, cand_scoress = list(zip(*sample_batch))

    n_cands = [cds.shape[0] for cds in cand_featuress]

    # convert to numpy arrays
    cand_featuress = np.concatenate(cand_featuress, axis=0)
    cand_scoress = np.concatenate(cand_scoress, axis=0)
    n_cands = np.array(n_cands)
    best_actions = np.array(sample_actions)

    # convert to tensors
    cand_featuress = torch.as_tensor(cand_featuress, dtype=torch.float32)
    cand_scoress = torch.as_tensor(cand_scoress, dtype=torch.float32)
    n_cands = torch.as_tensor(n_cands, dtype=torch.int32)
    best_actions = torch.as_tensor(sample_actions, dtype=torch.long)

    return cand_featuress, n_cands, best_actions, cand_scoress


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

        self.output_module = nn.Sequential(
            nn.Linear(self.n_input_feats, 1, bias=True),
        )
        
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


def process(policy, data_loader, DEVICE='cuda', optimizer=None):
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
    NB_EPOCHS = 3
    PATIENCE = 10
    EARLY_STOPPING = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PROBLEM = 'setcover'

    Path('examples/log/').mkdir(exist_ok=True)
    log = Logger(filename='examples/log/train_log_MLP.txt')

    sample_files = [str(path) for path in Path(f'examples/data/samples/{PROBLEM}/mlp/train').glob('sample_*.pkl')]
    train_files = sample_files[:int(0.8*len(sample_files))]
    valid_files = sample_files[int(0.8*len(sample_files)):]

    train_data = MLPDataset(train_files)
    #train_loader = torch.data.DataLoader(train_data, batch_size=32, shuffle=True)
    valid_data = MLPDataset(valid_files)
    #valid_loader = torch.data.DataLoader(valid_data, batch_size=128, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=32,
        shuffle=True, collate_fn=load_batch)
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=128,
        shuffle=True, collate_fn=load_batch)

    policy = MLPPolicy().to(DEVICE)

    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    for epoch in range(NB_EPOCHS):
        log(f"Epoch {epoch+1}")
        
        train_loss, train_acc = process(policy, train_loader, DEVICE, optimizer)
        log(f"Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}")

        valid_loss, valid_acc = process(policy, valid_loader, DEVICE, None)
        log(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}")

    torch.save(policy.state_dict(), f'examples/models/mlp_trained_params_{PROBLEM}.pkl')
