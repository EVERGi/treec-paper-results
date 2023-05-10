import pickle
from jupyter_client import protocol_version
from treec.prune_tree import prune_tree
from treec.train import find_best_individual, get_treestruct

from boptest.evaluation import evaluate, evaluate_trees, evaluate_with_leafs
from boptest.input_function import observation_to_input_function
from boptest.logger import BopLogger

# TODO figure out how this works


import requests
from collections import OrderedDict

from deap import base
from deap import creator
from deap import tools
from deap import cma
from deap import algorithms

import numpy as np
import random
import os
import sys

from treec.visualise_tree import display_binarytree

bop_gym_path = os.path.dirname(os.path.realpath(__file__)) + "/project1-boptest-gym/"
sys.path.insert(0, bop_gym_path)

from boptestGymEnv import BoptestGymEnv


class BoptestGymEnvCustomReward(BoptestGymEnv):
    """Define a custom reward for this building"""

    def compute_reward(self):
        """Custom reward function"""

        # Compute BOPTEST core kpis
        kpis = requests.get("{0}/kpi".format(self.url)).json()
        self.last_kpis = kpis

        # Calculate objective integrand function at this point
        objective_integrand = kpis["cost_tot"] * 12.0 * 16.0 + 100 * kpis["tdis_tot"]

        # print(f"Run KPIs {kpis}")
        # Compute reward
        reward = -(objective_integrand - self.objective_integrand)

        self.objective_integrand = objective_integrand

        return reward

    def reset(self):
        """
        Method to reset the environment. The associated building model is
        initialized by running the baseline controller for a
        `self.warmup_period` of time right before `self.start_time`.
        If `self.random_start_time` is True, a random time is assigned
        to `self.start_time` such that there are not episodes that overlap
        with the indicated `self.excluding_periods`. This is useful to
        define testing periods that should not use data from training.

        Returns
        -------
        observations: numpy array
            Reformatted observations that include measurements and
            predictions (if any) at the end of the initialization.

        """

        def find_start_time():
            """Recursive method to find a random start time out of
            `excluding_periods`. An episode and an excluding_period that
            are just touching each other are not considered as being
            overlapped.

            """
            start_time = random.randint(
                0 + self.bgn_year_margin, 3.1536e7 - self.end_year_margin
            )
            episode = (start_time, start_time + self.max_episode_length)
            if self.excluding_periods is not None:
                for period in self.excluding_periods:
                    if episode[0] < period[1] and period[0] < episode[1]:
                        # There is overlapping between episode and this period
                        # Try to find a good starting time again
                        start_time = find_start_time()
            # This point is reached only when a good starting point is found
            return start_time

            # if "time_period" not in self.scenario.keys():
            # Assign random start_time if it is None

        if self.random_start_time:
            self.start_time = find_start_time()
        # Initialize the building simulation
        res = requests.put(
            "{0}/initialize".format(self.url),
            data={
                "start_time": self.start_time,
                "warmup_period": self.warmup_period,
            },
        ).json()

        # else:
        #    res = requests.put(
        #        "{0}/initialize".format(self.url),
        #        data={},
        #    ).json()

        # Set simulation step
        requests.put("{0}/step".format(self.url), data={"step": self.step_period})

        # Set BOPTEST scenario
        requests.put("{0}/scenario".format(self.url), data=self.scenario)

        # Set forecasting parameters if predictive
        if self.is_predictive:
            forecast_parameters = {
                "horizon": self.predictive_period,
                "interval": self.step_period,
            }
            requests.put(
                "{0}/forecast_parameters".format(self.url), data=forecast_parameters
            )

        # Initialize objective integrand
        self.objective_integrand = 0.0

        # Get observations at the end of the initialization period
        observations = self.get_observations(res)

        self.episode_rewards = []

        return observations


def boptest_tree_train(common_params, algo_params, queue=None):
    input_func = common_params["input_func"]
    tot_steps = common_params["tot_steps_train"]
    log_folder = common_params["log_folder"]
    start_time_train = common_params["start_time_train"]
    case = common_params["case"]
    electricity_price = common_params["electricity_price"]
    port = common_params["port"]

    gen = algo_params["gen"]
    dimension = algo_params["dimension"]
    checkpoint = algo_params["checkpoint"]

    TreeStruct = get_treestruct(common_params, algo_params)

    # if queue is not None:
    #    queue.put(logger.folder_name)
    env = get_env_bop(
        case,
        start_time_train,
        render=False,
        scenario={"electricity_price": electricity_price},
        port=port,
    )

    tree_titles = [f"Action_{i}" for i in range(env.action_space.shape[0])]

    logger = BopLogger(log_folder, "tree", common_params, algo_params, tree_titles)

    params_evaluation = {
        "TreeStruct": TreeStruct,
        "input_func": input_func,
        "tot_steps": tot_steps,
        "env": env,
        "logger": logger,
    }

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    if checkpoint is None:
        toolbox = base.Toolbox()
        toolbox.register("evaluate", evaluate, params_evaluation=params_evaluation)
        strategy = cma.Strategy(centroid=[0.5] * dimension, sigma=0.5)

        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)
    else:
        file = open(checkpoint, "rb")
        toolbox = pickle.load(file)
        file.close()

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaGenerateUpdate(toolbox, ngen=gen, stats=stats, halloffame=hof)

    logger.save_checkpoint(toolbox)

    return logger.folder_name


def boptest_tree_validate(
    common_params, algo_params, folder_name, start_time_valid=None
):
    electricity_price = common_params["electricity_price"]
    time_period = common_params["time_period"]

    input_func = common_params["input_func"]
    tot_steps_train = common_params["tot_steps_train"]
    tot_steps_valid = common_params["tot_steps_valid"]
    case = common_params["case"]
    start_time_train = common_params["start_time_train"]
    port = common_params["port"]

    if start_time_valid is None:
        start_time_valid = common_params["start_time_valid"]

    logger = None

    TreeStruct = get_treestruct(common_params, algo_params)

    env = get_env_bop(
        case,
        start_time_train,
        render=True,
        scenario={"electricity_price": electricity_price},
        port=port,
    )

    params_prune = {
        "TreeStruct": TreeStruct,
        "input_func": input_func,
        "tot_steps": tot_steps_train,
        "env": env,
        "logger": logger,
    }
    indiv = find_best_individual(folder_name + "models/")
    trees = prune_individual(indiv, params_prune)

    env.render()

    tree_titles = [f"Action_{i}" for i in range(env.action_space.shape[0])]

    logger = BopLogger(
        folder_name + "validation/", "tree", common_params, algo_params, tree_titles
    )

    env = get_env_bop(
        case,
        start_time_valid,
        render=True,
        scenario={
            "electricity_price": electricity_price,
            "time_period": time_period,
        },
    )

    params_valid = {
        "TreeStruct": TreeStruct,
        "input_func": input_func,
        "tot_steps": tot_steps_valid,
        "env": env,
        "logger": logger,
    }

    result, _ = evaluate_trees(trees, params_valid)
    env.render()

    # kpis = requests.get("{0}/kpi".format(env.url)).json()
    # print(f"KPIs: {kpis}")
    print(env.last_kpis)
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
    _, leafs = evaluate_with_leafs(individual, params_prune, show_progress=True)
    for i, tree_raw in enumerate(trees):
        tree = TreeStruct(tree_raw, feature_names, [])
        tree.set_act_min_max(env.action_space.low[i], env.action_space.high[i])

        # Working temperature setting
        # tree.set_act_min_max(280.0, 310.0)

        leafs_tree = leafs[i]
        leafs_tree = prune_tree(tree, leafs_tree)

        trees_pruned.append(tree)
        if display:
            display_binarytree(tree, "Action_" + str(i), leafs_tree)

    return trees_pruned


def get_env_bop(
    case,
    start_time,
    render=False,
    scenario={"electricity_price": "highly_dynamic"},
    port="5000",
    tot_time_steps=100 * 24 * 12,
):
    url = f"http://127.0.0.1:{port}"

    start_time_tests = [(23 - 7) * 24 * 3600, (115 - 7) * 24 * 3600]

    episode_length_test = 14 * 24 * 3600
    warmup_period = 1 * 24 * 3600
    max_episode_length = 100 * 24 * 3600
    log_dir = None

    excluding_periods = []
    for start_time_test in start_time_tests:
        excluding_periods.append(
            (start_time_test, start_time_test + episode_length_test)
        )
    # Summer period (from June 21st till September 22nd).
    # Excluded since no heating during this period (nothing to learn).
    excluding_periods.append((173 * 24 * 3600, 266 * 24 * 3600))

    if case == "simple":
        env = BoptestGymEnvCustomReward(
            url=url,
            actions=["oveHeaPumY_u"],
            observations=OrderedDict([("reaTZon_y", (280.0, 310.0))]),
            random_start_time=False,
            start_time=start_time,
            excluding_periods=excluding_periods,
            max_episode_length=max_episode_length,
            warmup_period=warmup_period,
            step_period=900,
            render_episodes=render,
            log_dir=log_dir,
        )
    elif case == "A":
        env = BoptestGymEnvCustomReward(
            url=url,
            actions=["oveHeaPumY_u"],
            observations=OrderedDict(
                [
                    ("time", (0, 604800)),
                    ("reaTZon_y", (280.0, 310.0)),
                    ("PriceElectricPowerHighlyDynamic", (-0.4, 0.4)),
                ]
            ),
            scenario=scenario,
            predictive_period=0,
            start_time=start_time,
            random_start_time=False,
            excluding_periods=excluding_periods,
            max_episode_length=max_episode_length,
            warmup_period=warmup_period,
            step_period=900,
            render_episodes=render,
            log_dir=log_dir,
        )
    if case == "B":
        env = BoptestGymEnvCustomReward(
            url=url,
            actions=["oveHeaPumY_u"],
            observations=OrderedDict(
                [
                    ("time", (0, 604800)),
                    ("reaTZon_y", (280.0, 310.0)),
                    ("PriceElectricPowerHighlyDynamic", (-0.4, 0.4)),
                    ("LowerSetp[1]", (280.0, 310.0)),
                    ("UpperSetp[1]", (280.0, 310.0)),
                ]
            ),
            predictive_period=0,
            scenario=scenario,
            start_time=start_time,
            random_start_time=False,
            excluding_periods=excluding_periods,
            max_episode_length=max_episode_length,
            warmup_period=warmup_period,
            step_period=900,
            render_episodes=render,
            log_dir=log_dir,
        )
    if case == "C":
        env = BoptestGymEnvCustomReward(
            url=url,
            actions=["oveHeaPumY_u"],
            observations=OrderedDict(
                [
                    ("time", (0, 604800)),
                    ("reaTZon_y", (280.0, 310.0)),
                    ("PriceElectricPowerHighlyDynamic", (-0.4, 0.4)),
                    ("LowerSetp[1]", (280.0, 310.0)),
                    ("UpperSetp[1]", (280.0, 310.0)),
                ]
            ),
            predictive_period=3 * 3600,
            scenario=scenario,
            start_time=start_time,
            random_start_time=False,
            excluding_periods=excluding_periods,
            max_episode_length=max_episode_length,
            warmup_period=warmup_period,
            step_period=1800,
            render_episodes=render,
            log_dir=log_dir,
        )

    if case == "D":
        env = BoptestGymEnvCustomReward(
            url=url,
            actions=["oveHeaPumY_u"],
            observations=OrderedDict(
                [
                    ("time", (0, 604800)),
                    ("reaTZon_y", (280.0, 310.0)),
                    ("TDryBul", (265, 303)),
                    ("HDirNor", (0, 862)),
                    ("InternalGainsRad[1]", (0, 219)),
                    ("PriceElectricPowerHighlyDynamic", (-0.4, 0.4)),
                    ("LowerSetp[1]", (280.0, 310.0)),
                    ("UpperSetp[1]", (280.0, 310.0)),
                ]
            ),
            # predictive_period=24 * 3600,
            predictive_period=0,
            # regressive_period=6 * 3600,
            regressive_period=None,
            scenario=scenario,
            start_time=start_time,
            random_start_time=False,
            excluding_periods=excluding_periods,
            max_episode_length=max_episode_length,
            warmup_period=warmup_period,
            step_period=900,
            render_episodes=render,
            log_dir=log_dir,
        )
    if case == "E":
        max_episode_length = tot_time_steps * 900
        env = BoptestGymEnvCustomReward(
            url=url,
            actions=["oveHeaPumY_u"],
            observations=OrderedDict(
                [
                    ("time", (0, 604800)),
                    ("reaTZon_y", (280.0, 310.0)),
                    # ("TDryBul", (265, 303)),
                    # ("HDirNor", (0, 862)),
                    # ("InternalGainsRad[1]", (0, 219)),
                    ("PriceElectricPowerHighlyDynamic", (-0.4, 0.4)),
                    ("LowerSetp[1]", (280.0, 310.0)),
                    ("UpperSetp[1]", (280.0, 310.0)),
                ]
            ),
            # predictive_period=24 * 3600,
            predictive_period=0,
            # regressive_period=6 * 3600,
            regressive_period=None,
            scenario=scenario,
            start_time=start_time,
            random_start_time=False,
            excluding_periods=excluding_periods,
            max_episode_length=max_episode_length,
            warmup_period=warmup_period,
            step_period=900,
            render_episodes=render,
            log_dir=log_dir,
        )

    return env


def train_valid_function(port, electricity_price, time_period, num_gen):
    if time_period == "peak_heat_day":
        start_time_train = 1 * 24 * 3600
        start_time_valid = 16 * 24 * 3600
    else:
        start_time_train = (115 - 7 - 15) * 24 * 3600
        start_time_valid = (115 - 7) * 24 * 3600
    tot_steps_train = 14 * 24 * 4
    tot_steps_valid = 14 * 24 * 4

    common_params = {
        "input_func": observation_to_input_function,
        "tot_steps_train": tot_steps_train,
        "continuous": True,
        "case": "E",
        "tot_steps_valid": tot_steps_valid,
        "start_time_train": start_time_train,
        "start_time_valid": start_time_valid,
        "electricity_price": electricity_price,
        "time_period": time_period,
        # "train_time": 30 * 60,
        # "num_processes": num_proc,
        "log_folder": "boptest_results/",
        "port": port,
    }
    dimensions = 3 * 20 + 1
    algo_type = "tree"
    algo_params = {
        "gen": num_gen,
        "fixed": True,
        "dimension": dimensions,
        "checkpoint": None,
    }
    folder_name = boptest_tree_train(common_params, algo_params)
    print(folder_name)
    boptest_tree_validate(common_params, algo_params, folder_name)


def train_function_boptest(port, electricity_price, time_period, num_gen):
    if time_period == "peak_heat_day":
        start_time_train = 1 * 24 * 3600
        start_time_valid = 16 * 24 * 3600
    else:
        start_time_train = (115 - 7 - 15) * 24 * 3600
        start_time_valid = (115 - 7) * 24 * 3600
    tot_steps_train = 14 * 24 * 4
    tot_steps_valid = 14 * 24 * 4

    common_params = {
        "input_func": observation_to_input_function,
        "tot_steps_train": tot_steps_train,
        "continuous": True,
        "case": "E",
        "tot_steps_valid": tot_steps_valid,
        "start_time_train": start_time_train,
        "start_time_valid": start_time_valid,
        "electricity_price": electricity_price,
        "time_period": time_period,
        # "train_time": 30 * 60,
        # "num_processes": num_proc,
        "log_folder": "boptest_results/",
        "port": port,
    }
    dimensions = 3 * 20 + 1
    algo_params = {
        "gen": num_gen,
        "fixed": True,
        "dimension": dimensions,
        "checkpoint": None,
    }
    folder_name = boptest_tree_train(common_params, algo_params)
    return folder_name


if __name__ == "__main__":
    train_valid_function("5000", "highly_dynamic", "peak_heat_day", 150)
