
import ecole as ec
# import skopt
import numpy as np
# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from itertools import count


class Model(torch.nn.Module):
    def initialize_parameters(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight.data, gain=1)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias.data, 0)

    def save_state(self, filepath):
        torch.save(self.state_dict(), filepath)

    def restore_state(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))


class Policy(Model):
    def __init__(self):
        super().__init__()

        self.n_input_feats = 92
        self.ff_size = 256

        self.activation = torch.nn.ReLU()

        self.output_module = nn.Sequential(
            nn.Linear(self.n_input_feats, 1, bias=True),
        )

        # print(self.output_module)
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



def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    gamma = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


# -- ENVIRONMENT -- #

env = ec.environment.Branching(

    # set up a few SCIP parameters
    scip_params={
        "branching/scorefunc": 's',  # sum score function TODO: Change
        "branching/vanillafullstrong/priority": 666666,  
        # use vanillafullstrong (highest priority)
        "presolving/maxrounds": 0,  # deactivate presolving
        'limits/time': 3600
    },

    # observe
    observation_function=ec.observation.Khalil2016(),

    # minimize the total number of nodes
    reward_function=-ec.reward.NNodes(),

    # collect additional metrics for information purposes
    information_function={
        'nnodes': ec.reward.NNodes().cumsum(),
        'lpiters': ec.reward.LpIterations().cumsum(),
        'time': ec.reward.SolvingTime().cumsum(),
    }
)

# infinite instance generator, new instances will be generated on-the-fly
instances = ec.instance.CombinatorialAuctionGenerator(
    n_items=100, n_bids=100, add_item_prob=0.7)


# change those values as desired
n_iters = 100


# make the training process deterministic
seed = 42

env.seed(seed)  # environment (SCIP)
instances.seed(seed)  # instance generator
rng = np.random.RandomState(seed)  # optimizer


# -- TRAIN -- #


LEARNING_RATE = 0.001
NB_EPOCHS = 50
PATIENCE = 10
EARLY_STOPPING = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model()  # .to(DEVICE)
policy = Policy()  # .to(DEVICE)

"""observation = instances[0].to(DEVICE)

logits = policy(observation.constraint_features,
                observation.edge_index,
                observation.edge_attr,
                observation.variable_features)
action_distribution = F.softmax(logits[observation.candidates], dim=-1)

print(action_distribution)"""

optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


log_interval = 10
running_reward = 10
for i_episode in count(1):
    observation, action_set, reward_offset, done, info = env.reset(next(instances))
    ep_reward = 0
    for t in range(1, 10000):  # Don't infinite loop while learning
        # print(observation)
        
        """node_observation = (observation.row_features,
                            (observation.edge_features.indices,
                             observation.edge_features.values),
                            observation.column_features)
        node_observation = observation.row_features  # .to(DEVICE)"""
        node_observation = observation
        print(node_observation.shape)
        exit(0)
        
        #action = action_set[scores[action_set].argmax()]
        action = select_action(node_observation)
        observation, action_set, reward, done, info = env.step(action)
        policy.rewards.append(reward)
        ep_reward += reward
        if done:
            break

    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    finish_episode()
    if i_episode % log_interval == 0:
        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
    if running_reward > env.spec.reward_threshold:
        print(f"Solved! Running reward is now {running_reward} and "
              "the last episode runs to {t} time steps!")
        break

"""
# run the optimization
for i in tqdm(range(n_iters), ascii=True):

    # pick up a new random instance
    instance = next(instances)

    # start a new episode
    env.reset(instance)

    # get the next action from the optimizer
    x = opt.ask()
    action = {"branching/scorefac": x[0]}

    # apply the action and collect the reward
    _, _, reward, _, _ = env.step(action)

    # update the optimizer
    opt.tell(x, -reward)  # minimize the negated reward (eq. maximize the reward)

"""
# we set up more challenging instances
test_instances = ec.instance.CombinatorialAuctionGenerator(
    n_items=150, n_bids=750, add_item_prob=0.7)

seed = 1337

for policy in ('default', 'learned'):

    print(f"evaluating policy '{policy}'")
    results = []

    for i in range(5):

        # evaluate each policy in the exact same settings
        env.seed(seed+i)  # environment (SCIP)
        test_instances.seed(seed+i)  # instance generator

        # pick up the next instance
        instance = next(test_instances)

        # set up the episode initial state
        env.reset(instance)

        # get the action from the policy
        if policy == 'default':
            action = {}  # will use the default value from SCIP
        else:
            action = {"branching/scorefac": best_value}

        # apply the action and collect the reward
        _, _, _, _, info = env.step(action)

        print(f"  instance {i+1}: {info['nnodes']} nodes, {info['lpiters']} lpiters, {info['time']} secs")

        results.append(info['nnodes'])

    print(f"  average performance: {np.mean(results)} nodes")
