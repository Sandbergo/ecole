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
from models.gnn import GraphDataset, GNNPolicy, process



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

    LEARNING_RATE = 0.001
    NB_EPOCHS = 10
    PATIENCE = 10
    EARLY_STOPPING = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PROBLEM = 'cauctions'

    Path('examples/log/').mkdir(exist_ok=True)
    log = Logger(filename='examples/log/train_log.txt')

    sample_files = [str(path) for path in Path('examples/data/samples/cauctions/train').glob('sample_*.pkl')]
    train_files = sample_files[:int(0.8*len(sample_files))]
    valid_files = sample_files[int(0.8*len(sample_files)):]

    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.data.DataLoader(train_data, batch_size=32, shuffle=True)
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=128, shuffle=False)

    policy = GNNPolicy().to(DEVICE)

    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    for epoch in range(NB_EPOCHS):
        log(f"Epoch {epoch+1}")
        
        train_loss, train_acc = process(policy, train_loader, optimizer)
        log(f"Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}")

        valid_loss, valid_acc = process(policy, valid_loader, None)
        log(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}")

    torch.save(policy.state_dict(), f'examples/models/gnn_trained_params_{PROBLEM}.pkl')
