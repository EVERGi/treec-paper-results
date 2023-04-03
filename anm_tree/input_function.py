def observation_to_input_function(env, obs):

    inputs = list()
    input_info = list()

    min_space = env.observation_space.low
    max_space = env.observation_space.high

    count = 0
    for val_type in env.state_values:
        for input in val_type[1]:
            input_info.append(
                [f"{val_type[0]}_{input}", [min_space[count], max_space[count]]]
            )
            inputs.append(obs[count])
            count += 1

    return inputs, input_info
