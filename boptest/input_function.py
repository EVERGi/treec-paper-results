def observation_to_input_function(env, obs):

    inputs = list()
    input_info = list()

    min_space = env.observation_space.low
    max_space = env.observation_space.high

    for i, obs_name in enumerate(env.observations):

        input_info.append([f"{obs_name}", [min_space[i], max_space[i]]])
        inputs.append(obs[i])

    return inputs, input_info
