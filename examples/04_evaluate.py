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



def process(model, dataloader, top_k):
    """
    Executes only a forward pass of model over the dataset and computes accuracy
    Parameters
    ----------
    model : model.BaseModel
        A base model, which may contain some model.PreNormLayer layers.
    dataloader : torch.utils.data.DataLoader
        Dataset to use for training the model.
    top_k : list
        list of `k` (int) to estimate for accuracy using these many candidates
    Return
    ------
    mean_kacc : np.array
        computed accuracy for `top_k` candidates
    """

    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    for batch in dataloader:
        cand_features, n_cands, best_cands, cand_scores, weights  = map(lambda x:x.to(device), batch)
        batched_states = (cand_features)
        batch_size = n_cands.shape[0]
        weights /= batch_size # sum loss

        with torch.no_grad():
            logits = model(batched_states)  # eval mode
            logits = model.pad_output(logits, n_cands)  # apply padding now

        true_scores = model.pad_output(torch.reshape(cand_scores, (1, -1)), n_cands)
        true_bestscore = torch.max(true_scores, dim=-1, keepdims=True).values
        true_scores = true_scores.cpu().numpy()
        true_bestscore = true_bestscore.cpu().numpy()

        kacc = []
        for k in top_k:
            pred_top_k = torch.topk(logits, k=k).indices.cpu().numpy()
            pred_top_k_true_scores = np.take_along_axis(true_scores, pred_top_k, axis=1)
            kacc.append(np.mean(np.any(pred_top_k_true_scores == true_bestscore, axis=1)))
        kacc = np.asarray(kacc)

        mean_kacc += kacc * batch_size
        n_samples_processed += batch_size

    mean_kacc /= n_samples_processed

    return mean_kacc


if __name__ == "__main__":

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Path('examples/log/').mkdir(exist_ok=True)
    log = Logger(filename='examples/log/kacc_log.txt')

    policy = GNNPolicy().to(DEVICE)

    # observation = train_data[0].to(DEVICE)

    # logits = policy(observation.constraint_features, observation.edge_index, observation.edge_attr, observation.variable_features)
    # action_distribution = F.softmax(logits[observation.candidates], dim=-1)

    # print(action_distribution)


        test_loss, test_acc = process(model=policy, dataloader=valid_loader, topk=range(1,11))
        log(f"Test loss: {test_loss:0.3f}, accuracy {valid_acc:0.3f}")


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
