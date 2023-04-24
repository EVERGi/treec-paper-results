import csv
from anm6easy.evaluation import evaluate_trees
from anm6easy.input_function import observation_to_input_function
from anm6easy.train import prune_individual
from treec.train import get_treestruct
import gym

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

def csv_to_dict(filepath):

    with open(filepath, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        mydict = {rows[0]: rows[1] for rows in csvreader if len(rows) == 2}
    return mydict

def prune_tree(params_path, model_path):
    parameters = csv_to_dict(params_path)

    input_func = observation_to_input_function
    tot_steps_train = int(parameters["tot_steps_train"])
    gym_env = parameters["gym_env"]
    seed_train = int(parameters["seed_train"])

    TreeStruct = get_treestruct(parameters, parameters)

    env = gym.make(gym_env)

    params_prune = {
        "TreeStruct": TreeStruct,
        "input_func": input_func,
        "tot_steps": tot_steps_train,
        "seed": seed_train,
        "env": env,
        "logger": None,
    }
    file = open(model_path, "r")

    indiv_str_list = file.read().split(",")
    file.close()

    indiv = [float(i) for i in indiv_str_list]

    trees = prune_individual(indiv, params_prune)

    return trees

def visualise_trees_anm(params_path, model_path, seed_valid):

    print("Start pruning")
    trees = prune_tree(params_path, model_path)
    print("Pruning eneded")

    parameters = csv_to_dict(params_path)
    tot_steps_valid = int(parameters["tot_steps_valid"])
    gym_env = parameters["gym_env"]



    TreeStruct = get_treestruct(parameters, parameters)
    
    env = gym.make(gym_env)

    params_valid = {
        "TreeStruct": TreeStruct,
        "input_func": observation_to_input_function,
        "tot_steps": tot_steps_valid,
        "seed": seed_valid,
        "env": env,
        "logger": None,
    }
    print("Start validation")
    result, _ = evaluate_trees(trees, params_valid)
    print("Validation score: ", result)

    env = gym.make(gym_env)

    params_valid = {
        "TreeStruct": TreeStruct,
        "input_func": observation_to_input_function,
        "tot_steps": tot_steps_valid,
        "seed": seed_valid,
        "env": env,
        "logger": None,
    }

    result, _ = evaluate_trees(trees, params_valid, render=True)

if __name__ == "__main__":
    params_path = "anm6easy_paper_trees/ANM6Easy-v0_tree_0/params_run.csv"
    model_path = "anm6easy_paper_trees/ANM6Easy-v0_tree_0/tree_model.txt"

    visualise_trees_anm(params_path, model_path, 1)
