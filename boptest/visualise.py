from boptest.evaluation import evaluate_trees
from boptest.input_function import observation_to_input_function
from boptest.train import get_env_bop, prune_individual
from boptest.validate import csv_to_dict
from treec.train import get_treestruct

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 


import matplotlib.pyplot as plt

def prune_tree(params_path, model_path):
    parameters = csv_to_dict(params_path)
    time_period = parameters["time_period"]
    electricity_price = parameters["electricity_price"]
    port = "5000"

    if time_period == "peak_heat_day":
        start_time_train = 1 * 24 * 3600
        start_time_valid = 16 * 24 * 3600
    else:
        start_time_train = (115 - 7 - 15) * 24 * 3600
        start_time_valid = (115 - 7) * 24 * 3600
    tot_steps_train = 14 * 24 * 4
    #TODO change ste train
    # tot_steps_train = 2 * 4
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
        "gen": 150,
        "fixed": True,
        "dimension": dimensions,
        "checkpoint": None,
    }
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

    TreeStruct = get_treestruct(common_params, algo_params)

    env = get_env_bop(
        case,
        start_time_train,
        render=False,
        scenario={"electricity_price": electricity_price},
        port=port,
        tot_time_steps=tot_steps_train
    )

    params_prune = {
        "TreeStruct": TreeStruct,
        "input_func": input_func,
        "tot_steps": tot_steps_train,
        "env": env,
        "logger": None,
    }

    file = open(model_path, "r")

    indiv_str_list = file.read().split(",")
    file.close()

    indiv = [float(i) for i in indiv_str_list]
    trees = prune_individual(indiv, params_prune)

    return trees, common_params, params_prune

def visualise_trees_bop(params_path, model_path, time_period, electricity_price):
    trees, common_params, params_prune = prune_tree(params_path, model_path)
    
    case = common_params["case"]
    
    if time_period == "peak_heat_day":
        start_time_valid = 16 * 24 * 3600
    else:
        start_time_valid = (115 - 7) * 24 * 3600
    
    TreeStruct = params_prune["TreeStruct"]
    input_func = params_prune["input_func"]
    tot_steps_valid = common_params["tot_steps_valid"]

    env = get_env_bop(
        case,
        start_time_valid,
        render=False,
        scenario={
            "electricity_price": electricity_price,
            "time_period": time_period,
        },
        tot_time_steps=tot_steps_valid
    )

    params_valid = {
        "TreeStruct": TreeStruct,
        "input_func": input_func,
        "tot_steps": tot_steps_valid,
        "env": env,
        "logger": None,
    }

    result, _ = evaluate_trees(trees, params_valid)
    env.render()

    total_discomfort = env.last_kpis["tdis_tot"]
    total_cost = env.last_kpis["cost_tot"]
    print(f"Total discomfort: {total_discomfort}")
    print(f"Total operational cost: {total_cost}")
    plt.show(block=True)
    

if __name__ == "__main__":
    visualise_trees_bop("boptest_paper_trees/case_E_tree_0/params_run.csv", "boptest_paper_trees/case_E_tree_0/tree_model.txt", "peak_heat_day", "dynamic")