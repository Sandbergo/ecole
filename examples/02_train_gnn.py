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


if __name__ == "__main__":

    LEARNING_RATE = 0.001
    NB_EPOCHS = 10
    PATIENCE = 10
    EARLY_STOPPING = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PROBLEM = 'setcover'

    Path('examples/log/').mkdir(exist_ok=True)
    log = Logger(filename='examples/log/train_log.txt')
    sample_files = [str(path) for path in Path(f'examples/data/samples/{PROBLEM}/train').glob('sample_*.pkl')]
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
        
        train_loss, train_acc = process(policy, train_loader, DEVICE, optimizer)
        log(f"Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}")

        valid_loss, valid_acc = process(policy, valid_loader, DEVICE, None)
        log(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}")

    torch.save(policy.state_dict(), f'examples/models/gnn_trained_params_{PROBLEM}.pkl')
