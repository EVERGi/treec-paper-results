from treec.norm_func import (
    denormalise_input,
    normalise_input,
)

from .input_function import observation_to_input_function
import time

import numpy as np

from tqdm import tqdm


def evaluate_trees(
    trees, params_evaluation, individual=None, render=False, show_progress=False
):
    env = params_evaluation["env"]
    tot_steps = params_evaluation["tot_steps"]
    logger = params_evaluation["logger"]
    seed = params_evaluation["seed"]

    all_nodes_visited = [[] for _ in trees]

    env.seed(seed)
    obs = env.reset()

    _, input_info = observation_to_input_function(env, obs)

    obs_low = [i[1][0] for i in input_info]
    obs_high = [i[1][1] for i in input_info]

    score = 0
    tot_reward = 0
    for t in tqdm(range(tot_steps)) if show_progress else range(tot_steps):
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
            node_index = tree.node_stack.index(node)

            actions.append(action)
            all_nodes_visited[j].append(node_index)

        obs, r, _, _ = env.step(np.array(actions))
        score += env.gamma**t * r
        tot_reward += r
        if render:
            env.render()
            time.sleep(0.5)

    if logger is not None:
        new_best_tree = logger.episode_eval_log(individual, score)
        if new_best_tree:
            logger.save_tree_dot(trees, all_nodes_visited, score)

    return score, all_nodes_visited


def evaluate_with_leafs(individual, params_evaluation, show_progress=False):
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
        trees.append(tree)

    result, all_nodes_visited = evaluate_trees(
        trees, params_evaluation, individual, show_progress=show_progress
    )

    return result, all_nodes_visited


def evaluate(individual, params_evaluation):
    result, _ = evaluate_with_leafs(individual, params_evaluation)
    return (result,)
