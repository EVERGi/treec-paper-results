from treec.prune_tree import prune_tree
from treec.train import find_best_individual, get_treestruct
from evaluation import evaluate, evaluate_trees, evaluate_with_leafs
from input_function import observation_to_input_function
from logger import AnmLogger

from deap import base
from deap import creator
from deap import tools
from deap import cma
from deap import algorithms

import numpy as np

import gym
import gym_anm

from treec.visualise_tree import display_binarytree


def anm_tree_train(common_params, algo_params, queue=None):
    input_func = common_params["input_func"]
    tot_steps = common_params["tot_steps_train"]
    log_folder = common_params["log_folder"]
    gym_env = common_params["gym_env"]
    seed_train = common_params["seed_train"]

    gen = algo_params["gen"]
    dimension = algo_params["dimension"]

    TreeStruct = get_treestruct(common_params, algo_params)

    # if queue is not None:
    #    queue.put(logger.folder_name)
    env = gym.make(gym_env)

    tree_titles = [f"Action_{i}" for i in range(env.action_space.shape[0])]

    logger = AnmLogger(log_folder, "tree", common_params, algo_params, tree_titles)

    params_evaluation = {
        "TreeStruct": TreeStruct,
        "input_func": input_func,
        "tot_steps": tot_steps,
        "seed": seed_train,
        "env": env,
        "logger": logger,
    }

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate, params_evaluation=params_evaluation)
    strategy = cma.Strategy(centroid=[0.5] * dimension, sigma=0.5)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaGenerateUpdate(toolbox, ngen=gen, stats=stats, halloffame=hof)

    return logger.folder_name


def anm_tree_validate(common_params, algo_params, folder_name, seed_valid=None):

    input_func = common_params["input_func"]
    tot_steps_train = common_params["tot_steps_train"]
    tot_steps_valid = common_params["tot_steps_valid"]
    gym_env = common_params["gym_env"]
    seed_train = common_params["seed_train"]
    if seed_valid is None:
        seed_valid = common_params["seed_valid"]

    logger = None

    TreeStruct = get_treestruct(common_params, algo_params)

    env = gym.make(gym_env)

    params_prune = {
        "TreeStruct": TreeStruct,
        "input_func": input_func,
        "tot_steps": tot_steps_train,
        "seed": seed_train,
        "env": env,
        "logger": logger,
    }
    indiv = find_best_individual(folder_name + "models/")
    trees = prune_individual(indiv, params_prune)

    tree_titles = [f"Action_{i}" for i in range(env.action_space.shape[0])]

    logger = AnmLogger(
        folder_name + "validation/", "tree", common_params, algo_params, tree_titles
    )

    params_valid = {
        "TreeStruct": TreeStruct,
        "input_func": input_func,
        "tot_steps": tot_steps_valid,
        "seed": seed_valid,
        "env": env,
        "logger": logger,
    }

    result, _ = evaluate_trees(trees, params_valid)

    print("Validation score: ", result)

    return result


def prune_individual(individual, params_prune, display=False):

    input_func = params_prune["input_func"]
    TreeStruct = params_prune["TreeStruct"]
    env = params_prune["env"]

    _, feature_names = input_func(env, [0] * env.observation_space.shape[0])

    num_actions = env.action_space.shape[0]
    actions_names = []
    trees = list()
    slice_length = len(individual) // num_actions

    for i in range(num_actions):
        slice_tree = individual[i * slice_length : (i + 1) * slice_length]
        trees.append(slice_tree)

    trees_pruned = list()

    _, feature_names = input_func(env, [0] * env.observation_space.shape[0])
    _, leafs = evaluate_with_leafs(individual, params_prune)
    for i, tree_raw in enumerate(trees):
        tree = TreeStruct(tree_raw, feature_names, [])
        tree.set_act_min_max(env.action_space.low[i], env.action_space.high[i])

        leafs_tree = leafs[i]
        leafs_tree = prune_tree(tree, leafs_tree)

        trees_pruned.append(tree)
        if display:
            display_binarytree(tree, "Action_" + str(i), leafs_tree)

    return trees_pruned

def train_function_anm6easy(seed, train_gen):
    common_params = {
        "input_func": observation_to_input_function,
        "tot_steps_train": 300,
        "continuous": True,
        "gym_env": "ANM6Easy-v0",
        "tot_steps_valid": 3000,
        "seed_train": seed,
        "seed_valid": 100,
        # "train_time": 30 * 60,
        # "num_processes": num_proc,
        "log_folder": "anm6easy_results/",
    }
    dimensions = (3 * 20 + 1) * 6
    algo_type = "tree"
    algo_params = {
        "gen": train_gen,
        "fixed": True,
        "dimension": dimensions,
    }

    folder_name = anm_tree_train(common_params, algo_params)
    return folder_name


if __name__ == "__main__":
    common_params = {
        "input_func": observation_to_input_function,
        "tot_steps_train": 300,
        "continuous": True,
        "gym_env": "ANM6Easy-v0",
        "tot_steps_valid": 3000,
        "seed_train": 0,
        "seed_valid": 100,
        # "train_time": 30 * 60,
        # "num_processes": num_proc,
        "log_folder": "anm6easy_results/",
    }
    num_gen = 1500
    dimensions = (3 * 20 + 1) * 6
    algo_type = "tree"
    algo_params = {
        "gen": num_gen,
        "fixed": True,
        "dimension": dimensions,
    }

    folder_name = anm_tree_train(common_params, algo_params)
    print(folder_name)

    anm_tree_validate(common_params, algo_params, folder_name)
