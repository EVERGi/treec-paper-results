import pickle
from boptest.input_function import observation_to_input_function
from boptest.train import boptest_tree_train, boptest_tree_validate
import numpy as np
from deap import creator, base, tools, cma, algorithms

import matplotlib.pyplot as plt

from boptest.train import BoptestGymEnvCustomReward


def evaluate(individual):
    return [sum(individual)]


if __name__ == "__main__":
    num_gen = 1
    electricity_price = "highly_dynamic"
    time_period = "peak_heat_day"
    port = "5000"
    if time_period == "peak_heat_day":
        start_time_train = 1 * 24 * 3600
        start_time_valid = 16 * 24 * 3600
    else:
        start_time_train = (115 - 7 - 15) * 24 * 3600
        start_time_valid = (115 - 7) * 24 * 3600
    tot_steps_train = 1 * 24 * 4
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
        "checkpoint": "boptest_results/case_E_tree_1/checkpoint/toolbox.pkl",
    }

    folder_name = boptest_tree_train(common_params, algo_params)
