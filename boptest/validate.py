from boptest.input_function import observation_to_input_function
from boptest.train import boptest_tree_validate
import csv


def csv_to_dict(filepath):

    with open(filepath, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        mydict = {rows[0]: rows[1] for rows in csvreader if len(rows) == 2}
    return mydict


def validate_model(folder_name):
    parameters = csv_to_dict(f"{folder_name}/params_run.csv")
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
        "gen": 100,
        "fixed": True,
        "dimension": dimensions,
        "checkpoint": None,
    }
    print(folder_name)
    boptest_tree_validate(common_params, algo_params, folder_name)


if __name__ == "__main__":
    validate_model("boptest_results/case_E_tree_10/")
