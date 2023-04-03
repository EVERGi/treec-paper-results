import gym
import time
from gym_anm import MPCAgentPerfect, MPCAgentConstant
import numpy as np
from .input_function import observation_to_input_function
from .train import prune_individual
from treec.norm_func import denormalise_input, normalise_input
from treec.train import find_best_individual
from treec.tree import BinaryTreeFixedCont


def run(seed):
    env = gym.make("gym_anm:ANM6Easy-v0")
    env.seed(seed)
    o = env.reset()

    reward_sum = 0
    history_obervation = [[] for _ in range(env.state_N)]
    for t in range(3000):
        a = env.action_space.sample()
        # print(a)
        o, r, done, info = env.step(a)
        # print(o)
        # print(env.state_values)
        # env.render()
        # time.sleep(1)
        for i, val in enumerate(o):
            history_obervation[i].append(val)

        # print(f"t={t}, r_t={r:.3}")

        reward_sum += r * env.gamma**t
        if done:
            break
        if t == 2998:
            print("Wowowoow")
        if t == 2999:
            print("Wowowoow")
        # print(reward_sum)

    # reward_sum /= 5
    # print(reward_sum)
    # env.close()
    state_val = list()
    for val_type in env.state_values:
        for input in val_type[1]:
            state_val += [f"{val_type[0]}_{input}"]

    # for i, val_list in enumerate(history_obervation):
    # print(state_val[i])
    # print(f"Obs {i}")
    # print(f"Min: {min(val_list)}")
    # print(f"Max: {max(val_list)}")
    # print(env.P_maxs)

    # env.close()
    return t, reward_sum


def run_MPC_per_32(seed=1):
    print("MPC_run")
    env = gym.make("ANM6Easy-v0")
    env.seed(seed)
    o = env.reset()

    # Initialize the MPC policy.
    agent = MPCAgentPerfect(
        env.simulator,
        env.action_space,
        env.gamma,
        safety_margin=0.94,
        planning_steps=32,
    )

    reward_sum = 0
    # Run the policy.
    for t in range(3000):
        a = agent.act(env)
        obs, r, done, _ = env.step(a)
        env.render()
        time.sleep(1)
        print(f"t={t}, r_t={r:.3}")
        reward_sum += r * env.gamma**t

    # reward_sum /= 5
    print(reward_sum)


def run_MPC_const_16():
    print("MPC_run")
    env = gym.make("ANM6Easy-v0")

    o = env.reset()

    # Initialize the MPC policy.
    agent = MPCAgentPerfect(
        env.simulator,
        env.action_space,
        env.gamma,
        safety_margin=0.92,
        planning_steps=16,
    )

    reward_sum = 0
    # Run the policy.
    for t in range(3000):
        a = agent.act(env)
        obs, r, done, _ = env.step(a)
        print(f"t={t}, r_t={r:.3}")
        reward_sum += r * env.gamma**2

    reward_sum /= 5
    print(reward_sum)


def run_tree(tree_folder):
    env = gym.make("gym_anm:ANM6Easy-v0")
    env.seed(3)

    TreeStruct = BinaryTreeFixedCont
    params_prune = {
        "TreeStruct": TreeStruct,
        "input_func": observation_to_input_function,
        "tot_steps": 200,
        "env": env,
        "logger": None,
        "seed": 3,
    }

    indiv = find_best_individual(tree_folder + "models/")
    trees = prune_individual(indiv, params_prune)

    env.seed(100)
    obs = env.reset()
    _, input_info = observation_to_input_function(env, obs)

    obs_low = [i[1][0] for i in input_info]
    obs_high = [i[1][1] for i in input_info]

    score = 0
    tot_reward = 0
    for t in range(3000):
        actions = list()

        norm_obs = np.array(
            [
                normalise_input(obs_i, obs_low[i], obs_high[i])
                for i, obs_i in enumerate(obs)
            ]
        )

        for j, tree in enumerate(trees):
            node = tree.get_action(norm_obs)
            low_bound = env.action_space.low[j]
            high_bound = env.action_space.high[j]
            action = denormalise_input(node.value, low_bound, high_bound)

            actions.append(action)

        obs, r, _, _ = env.step(np.array(actions))
        env.render()
        time.sleep(1)

        score += env.gamma**t * r
        tot_reward += r

    env.close()


if __name__ == "__main__":

    tot_time = 0
    tot_rew = 0
    for i in range(100):
        t, rew = run(i)
        tot_time += t / 100
        tot_rew += rew / 100

    print(tot_time)
    print(tot_rew)
    # folder = "paper_ANM6_compare/ANM6Easy-v0_tree_3/"

    # run_tree(folder)

    # run_MPC_const_16()
    # run_MPC_per_32(100)
