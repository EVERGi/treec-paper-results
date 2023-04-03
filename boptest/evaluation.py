from treec.norm_func import (
    denormalise_input,
    normalise_input,
)

from treec.visualise_tree import display_binarytree
from treec.prune_tree import prune_tree

from copy import deepcopy

from boptest.input_function import observation_to_input_function

import numpy as np
import math

from matplotlib import pyplot as plt


def evaluate_trees(trees, params_evaluation, individual=None):
    env = params_evaluation["env"]
    tot_steps = params_evaluation["tot_steps"]
    logger = params_evaluation["logger"]

    all_nodes_visited = [[] for _ in trees]

    obs = env.reset()

    _, input_info = observation_to_input_function(env, obs)

    obs_low = [i[1][0] for i in input_info]
    obs_high = [i[1][1] for i in input_info]

    tot_reward = 0
    for t in range(tot_steps):
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
            action = math.floor(action * 11) / 10
            node_index = tree.node_stack.index(node)

            """
            # Working temperature setting
            low_bound = 280.0
            high_bound = 310.0
            node_val = denormalise_input(node.value, low_bound, high_bound)

            if node_val < obs[1]:
                action = 0.0
            else:
                action = 1.0
            """
            actions.append(action)
            all_nodes_visited[j].append(node_index)

        obs, r, _, _ = env.step(np.array(actions))
        tot_reward += r

    if logger is not None:
        new_best_tree = logger.episode_eval_log(individual, tot_reward)

        if new_best_tree:
            logger.save_tree_dot(trees, all_nodes_visited, tot_reward)
    # print(tot_reward)

    return tot_reward, all_nodes_visited


def evaluate_with_leafs(individual, params_evaluation):
    env = params_evaluation["env"]

    TreeStruct = params_evaluation["TreeStruct"]
    input_func = params_evaluation["input_func"]

    for i, indiv in enumerate(individual):
        if indiv < 0:
            individual[i] = 0
        elif indiv >= 1:
            individual[i] = 0.9999

    num_actions = env.action_space.shape[0]

    _, feature_names = input_func(env, [0] * env.observation_space.shape[0])

    actions_names = []
    trees = list()
    slice_length = len(individual) // num_actions
    for i in range(num_actions):
        slice_tree = individual[i * slice_length : (i + 1) * slice_length]
        tree = TreeStruct(slice_tree, feature_names, actions_names)
        tree.set_act_min_max(env.action_space.low[i], env.action_space.high[i])

        # Working temperature setting
        # tree.set_act_min_max(280.0, 310.0)

        trees.append(tree)

    result, all_nodes_visited = evaluate_trees(trees, params_evaluation, individual)

    return result, all_nodes_visited


def evaluate(individual, params_evaluation):
    result, _ = evaluate_with_leafs(individual, params_evaluation)
    return (result,)
