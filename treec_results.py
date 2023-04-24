import argparse
from boptest.train import train_function_boptest
from anm6easy.train import train_function_anm6easy

from boptest.visualise import visualise_trees_bop
from anm6easy.visualise import visualise_trees_anm
import os


def anm6easy_train(scenario, train_gen):
    folder_path = train_function_anm6easy(scenario, train_gen)

    return folder_path

def boptest_train(scenario, train_gen):
    time_period, electricity_price = bop_scenarios(scenario)
    
    folder_path = train_function_boptest("5000", electricity_price, time_period, train_gen)
    return folder_path

def anm6easy_visu(scenario, path_tree):
    params_path = get_params_file(path_tree)

    visualise_trees_anm(params_path, path_tree, scenario)

def boptest_visu(scenario, path_tree):
    time_period, electricity_price = bop_scenarios(scenario)
    
    params_path = get_params_file(path_tree)

    visualise_trees_bop(params_path, path_tree,time_period,electricity_price)

def get_params_file(path_tree):
    same_dir_params = "/".join(path_tree.split("/")[:-1])+"/params_run.csv"
    parent_dir_params = "/".join(path_tree.split("/")[:-2])+"/params_run.csv"
    if os.path.isfile(same_dir_params):
        params_path = same_dir_params
    elif os.path.isfile(parent_dir_params):
        params_path = parent_dir_params
    else:
        raise argparse.ArgumentTypeError('No params_run.csv file in the same or parent directory of the path_tree argument given.')
    
    return params_path

def bop_scenarios(scenario):
    if scenario==0:
        time_period = "peak_heat_day"
        electricity_price = "constant"
    elif scenario == 1:
        time_period = "typical_heat_day"
        electricity_price = "constant"
    elif scenario == 2:
        time_period = "peak_heat_day"
        electricity_price = "dynamic"
    elif scenario == 3:
        time_period = "typical_heat_day"
        electricity_price = "dynamic"
    elif scenario == 4:
        time_period = "peak_heat_day"
        electricity_price = "highly_dynamic"
    elif scenario == 5:
        time_period = "typical_heat_day"
        electricity_price = "highly_dynamic"
    else:
        raise argparse.ArgumentTypeError('Scenario should be an integer between 0 and 5 for the boptest case.')

    return time_period, electricity_price

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute the trainings of TreeC of visualise the trees obtained for the results in the paper.')


    parser.add_argument('-c','--case', type=str, choices=["anm", "bop"], 
                        help='Select anm6easy (a) or botpest case (b).', required=True)
    parser.add_argument('-m','--mode', type=str, choices=["train", "visu"],
                        help='Choose train or visualisation mode.', required=True)
    parser.add_argument('-s','--scenario', type=int,
                        help='Which scenario of the study cases to use, seed for anm6easy or int from 0 to 5 for the 6 different boptest scenarios.', required=True)
    parser.add_argument('-p','--path_tree', type=str,
                        help='Path to tree model to visualise. (default: Tree visualised in paper)', required=False, default="default")
    parser.add_argument('-t', '--train_gen', type=int,
                        help='Number of generations for the training. (default: Number of generations in paper)', required=False, default="default")
    
    args = parser.parse_args()

    if args.mode == "train":
        if args.case == "anm":
            folder_path = anm6easy_train(args.scenario, args.train_gen)
        elif args.case == "bop":
            folder_path = boptest_train(args.scenario, args.train_gen)
        print("Training finnished.")
        print("Results in:")
        print(folder_path)
    elif args.mode == "visu":
        if args.case == "anm":
            anm6easy_visu(args.scenario, args.path_tree)
        elif args.case == "bop":
            boptest_visu(args.scenario, args.path_tree)