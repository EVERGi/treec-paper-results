import gym
import time
import numpy as np

# Don't show the warnings
import warnings

warnings.filterwarnings("ignore")


def random_run(seed):
    env = gym.make("gym_anm:ANM6Easy-v0")
    env.seed(seed)
    o = env.reset()

    reward_sum = 0
    history_obervation = [[] for _ in range(env.state_N)]
    for t in range(3000):
        a = env.action_space.sample()
        # print(a)
        o, r, done, info = env.step(a)
        # env.render()
        # time.sleep(1)
        for i, val in enumerate(o):
            history_obervation[i].append(val)
        reward_sum += r * env.gamma**t
        if done:
            break_step = t
            break

    env.close()
    return reward_sum, break_step


def average_random_score():
    scores = []
    steps = []
    for seed in range(10):
        score, break_step = random_run(seed)
        scores.append(score)
        steps.append(break_step)
    average_score = np.mean(scores)
    average_step = np.mean(steps)
    return average_score, average_step


def calc_score_step_avg_std():
    avg_scores = []
    avg_steps = []
    for i in range(20):
        avg_score, avg_step = average_random_score()
        avg_scores.append(avg_score)
        avg_steps.append(avg_step)
        print(avg_score)
        print(avg_step)

    print("Average score: ", np.mean(avg_scores))
    print("Standard deviation: ", np.std(avg_scores))
    print("Average step: ", np.mean(avg_steps))


if __name__ == "__main__":
    calc_score_step_avg_std()
