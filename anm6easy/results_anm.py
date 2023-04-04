from .input_function import observation_to_input_function
from .train import anm_tree_train, anm_tree_validate
import multiprocessing


def train_and_validate(seed_train):
    seed_valid = 100
    common_params = {
        "input_func": observation_to_input_function,
        "tot_steps_train": 300,
        "continuous": True,
        "gym_env": "ANM6Easy-v0",
        "tot_steps_valid": 3000,
        "log_folder": "paper_ANM6_compare/",
        "seed_train": seed_train,
        "seed_valid": seed_valid,
    }
    num_gen = 1000
    dimensions = (3 * 20 + 1) * 6
    algo_params = {
        "gen": num_gen,
        "fixed": True,
        "dimension": dimensions,
    }

    folder_name = anm_tree_train(common_params, algo_params)
    print(folder_name)

    anm_tree_validate(common_params, algo_params, folder_name)


if __name__ == "__main__":
    jobs = list()
    for i in range(5):
        job = multiprocessing.Process(target=train_and_validate, args=(i,))
        job.start()
        jobs.append(job)

    for job in jobs:
        job.join()
