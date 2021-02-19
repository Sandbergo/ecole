
import ecole as ec
import skopt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

env = ec.environment.Configuring(

    # set up a few SCIP parameters
    scip_params={
        "branching/scorefunc": 's',  # sum score function
        "branching/vanillafullstrong/priority": 666666,  # use vanillafullstrong (highest priority)
        "presolving/maxrounds": 0,  # deactivate presolving
    },

    # pure bandit, no observation
    observation_function=None,

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
n_burnins = 10

# make the training process deterministic
seed = 42

env.seed(seed)  # environment (SCIP)
instances.seed(seed)  # instance generator
rng = np.random.RandomState(seed)  # optimizer

# set up the optimizer
opt = skopt.Optimizer(
    dimensions=[(0.0, 1.0)], base_estimator="GP", n_initial_points=n_burnins, random_state=rng,
    acq_func="PI", acq_optimizer="sampling", acq_optimizer_kwargs={'n_points': 10})

assert n_iters > n_burnins

# run the optimization
for i in range(n_iters):

    if (i+1) % 10 == 0:
        print(f"iteration {i+1} / {n_iters}")

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


model = opt.models[-1]

x = np.linspace(0, 1, 500)
x_model = opt.space.transform(x.reshape(-1, 1).tolist())

fig, ax1 = plt.subplots()

# points sampled during optimization
lns1 = ax1.plot(opt.Xi, opt.yi, "r.", markersize=8, label="Collected data")

# value function estimation
y_mean, y_std = model.predict(x_model, return_std=True)
lns2 = ax1.plot(x, y_mean, "g--", label=r"Value function")
ax1.fill_between(x, y_mean - 1.6 * y_std, y_mean + 1.6 * y_std,
                 alpha=0.2, fc="g", ec="None")

# probability of improvement estimation
x_pi = skopt.acquisition.gaussian_pi(x_model, model, y_opt=np.min(opt.yi))
ax2 = ax1.twinx()
lns3 = ax2.plot(x, x_pi, "b", label="Prob. of improvement")

ax1.set_title(f"Model obtained after {n_iters} iterations")
ax1.set_ylabel(f"number of nodes (neg. reward)")
ax1.set_xlabel(f"$branching/scorefac$ parameter value (action)")

ax2.set_ylabel(f"Probability value")

# Legend
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper center')

plt.savefig('test.png')

# get best value based on a grid search on the value function estimator
best_value = x[np.argmin(y_mean)]
print(f"Best parameter value: branching/scorefac = {best_value}")

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
