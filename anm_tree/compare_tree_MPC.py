import gym
import time
from gym_anm import MPCAgentPerfect
from anm_tree.input_function import observation_to_input_function
from anm_tree.train import anm_tree_validate
import csv

import multiprocessing

def csv_to_dict(filepath):

    with open(filepath, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        mydict = {rows[0]: rows[1] for rows in csvreader if len(rows) == 2}
    return mydict


def run_MPC(seed, result_file):
    env = gym.make("ANM6Easy-v0")
    env.seed(seed)
    o = env.reset()

    print(f"seed {seed}: st time {o[-1]}")
    # Initialize the MPC policy.
    agent = MPCAgentPerfect(
        env.simulator,
        env.action_space,
        env.gamma,
        safety_margin=0.94,
        planning_steps=32,
    )

    reward_sum = 0
    # Run the policy.
    for t in range(3000):
        a = agent.act(env)
        obs, r, done, _ = env.step(a)
        reward_sum += r * env.gamma**t

    print(reward_sum)

    file = open(result_file, "a+")
    file.write(f"MPC_32,{seed},{reward_sum}\n")
    file.close()

    return reward_sum


def run_tree(seed, folder, result_file=None):

    params_run = csv_to_dict(folder + "/params_run.csv")
    common_params = {
        "input_func": observation_to_input_function,
        "tot_steps_train": int(params_run["tot_steps_train"]),
        "continuous": bool(params_run["continuous"]),
        "gym_env": params_run["gym_env"],
        "tot_steps_valid": int(params_run["tot_steps_valid"]),
        "log_folder": params_run["log_folder"],
        "seed_train": int(params_run["seed_train"]),
        "seed_valid": seed,
    }

    algo_params = {
        "gen": int(params_run["gen"]),
        "fixed": bool(params_run["fixed"]),
        "dimension": int(params_run["dimension"]),
    }
    result = anm_tree_validate(common_params, algo_params, folder, seed)

    file = open(result_file, "a+")
    file.write(f"{folder},{seed},{result}\n")
    file.close()

    return result


if __name__ == "__main__":

    result_file = "paper_ANM6_compare/results.csv"
    jobs = list()
    for seed in range(100, 105):
        time.sleep(0.1)
        job = multiprocessing.Process(
            target=run_MPC,
            args=(
                seed,
                result_file,
            ),
        )
        job.start()
        jobs.append(job)
        for i in range(5):
            time.sleep(0.1)
            job = multiprocessing.Process(
                target=run_tree,
                args=(
                    seed,
                    f"paper_ANM6_compare/ANM6Easy-v0_tree_{i}/",
                    result_file,
                ),
            )
            job.start()
            jobs.append(job)

    for job in jobs:
        job.join()
