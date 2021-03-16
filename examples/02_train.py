import gzip
import pickle
import numpy as np
import ecole
import torch
import torch.nn.functional as F
import torch_geometric
import os
import argparse
from pathlib import Path

from utilities import Logger
from model_utilities import process
from data_utilities import GraphDataset
from models.mlp import MLPPolicy
from models.gnn import GNNPolicy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model',
        help='Model name.',
        choices=['gnn', 'mlp'],
    )
    parser.add_argument(
        '-p', '--problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=-1,
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    LEARNING_RATE = 0.001
    NB_EPOCHS = 10
    PATIENCE = 10
    EARLY_STOPPING = 20
    POLICY_DICT = {'gnn': GNNPolicy(), 'mlp': MLPPolicy()}
    PROBLEM = args.problem
    SEED = args.seed
    
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        DEVICE = torch.device("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = POLICY_DICT[args.model].to(DEVICE)

    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(rng.randint(np.iinfo(int).max))

    Path('examples/log/').mkdir(exist_ok=True)
    log = Logger(filename='examples/log/02_train')
    
    log(f'Model:   {args.model}')
    log(f'Problem: {PROBLEM}')
    log(f'Device:  {DEVICE}')
    log(f'Lr:      {LEARNING_RATE}')
    log(f'Epochs:  {NB_EPOCHS}')
    log(str(policy))


    # --- TRAIN --- #
    sample_files = [str(path) for path in Path(f'examples/data/samples/{PROBLEM}/train').glob('sample_*.pkl')]
    train_files = sample_files[:int(0.8*len(sample_files))]
    valid_files = sample_files[int(0.8*len(sample_files)):]

    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.data.DataLoader(train_data, batch_size=32, shuffle=False)
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=128, shuffle=False)

    log('Beginning training')
    valid_loss, valid_acc = process(policy=policy, data_loader=valid_loader, device=DEVICE, optimizer=None)
    log(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}")

    for epoch in range(NB_EPOCHS):
        log(f"Epoch {epoch+1}")
        
        train_loss, train_acc = process(policy=policy, data_loader=train_loader, device=DEVICE, optimizer=optimizer)
        log(f"Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}")

        valid_loss, valid_acc = process(policy=policy, data_loader=valid_loader, device=DEVICE, optimizer=None)
        log(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}")

    model_filename = f'examples/models/{args.model}_trained_params_{PROBLEM}.pkl'
    log(f'Saving model as {model_filename}')
    torch.save(policy.state_dict(), model_filename)
    log('End of training.')
