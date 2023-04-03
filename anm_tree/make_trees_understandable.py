import os
import re


def parse_folder_dot(tree_folder):
    for filename in os.listdir(tree_folder):
        file = open(tree_folder + filename, "r+")
        content = file.read()
        content = replace_action_names(content)

        content = replace_input_name_color(content)
        file.close()
        new_file = open(tree_folder + filename, "w+")
        new_file.write(content)
        new_file.close()


def replace_action_names(file_content):
    file_content = file_content.replace("Action_0", "Solar P value max")
    file_content = file_content.replace("Action_1", "Wind P value max")
    file_content = file_content.replace("Action_2", "Solar Q value")
    file_content = file_content.replace("Action_3", "Wind Q value")
    file_content = file_content.replace("Action_4", "Battery power P")
    file_content = file_content.replace("Action_5", "Battery Q value")

    return file_content


def replace_input_name_color(file_content):

    file_content = replace_start_end(
        file_content,
        "dev_p_1:\n",
        '" fillcolor=2',
        1,
        "Power residential:\n",
        '" fillcolor=2',
    )

    file_content = replace_start_end(
        file_content,
        "dev_p_2:\n",
        '" fillcolor=3',
        1,
        "Power PV:\n",
        '" fillcolor=3',
    )

    file_content = replace_start_end(
        file_content,
        "dev_p_3:\n",
        '" fillcolor=4',
        1,
        "Power Industry:\n",
        '" fillcolor=4',
    )
    file_content = replace_start_end(
        file_content,
        "dev_p_4:\n",
        '" fillcolor=5',
        1,
        "Power WT:\n",
        '" fillcolor=5',
    )
    file_content = replace_start_end(
        file_content,
        "dev_p_5:\n",
        '" fillcolor=6',
        1,
        "Power EV charger:\n",
        '" fillcolor=6',
    )

    file_content = replace_start_end(
        file_content,
        "dev_q_1:\n",
        '" fillcolor=9',
        5,
        "Power residential:\n",
        '" fillcolor=2',
    )
    file_content = replace_start_end(
        file_content,
        "dev_q_2:\n",
        '" fillcolor=10',
        5,
        "Power PV:\n",
        '" fillcolor=3',
    )

    file_content = replace_start_end(
        file_content,
        "dev_q_3:\n",
        '" fillcolor=11',
        5,
        "Power Industry:\n",
        '" fillcolor=4',
    )
    file_content = replace_start_end(
        file_content,
        "dev_q_5:\n",
        '" fillcolor=1',
        5,
        "Power EV charger:\n",
        '" fillcolor=6',
    )

    file_content = replace_start_end(
        file_content,
        "des_soc_6:\n",
        '" fillcolor=3',
        1,
        "SOC battery:\n",
        '" fillcolor=3',
    )

    file_content = replace_start_end(
        file_content,
        "gen_p_max_2:\n",
        '" fillcolor=4',
        1,
        "Max possible power PV:\n",
        '" fillcolor=4',
    )

    file_content = replace_start_end(
        file_content,
        "gen_p_max_4:\n",
        '" fillcolor=5',
        1,
        "Max possible power WT:\n",
        '" fillcolor=5',
    )
    file_content = replace_start_end(
        file_content,
        "aux_0:\n",
        '" fillcolor=6',
        1,
        "Time in 15 minutes:\n",
        '" fillcolor=6',
    )

    return file_content


def replace_start_end(file_content, start, end, mult, start_repl, end_repl):
    regex = start + "-?[0-9]+.[0-9]+" + end
    all_regex = re.findall(regex, file_content)
    for to_replace in all_regex:
        number = float(to_replace.replace(start, "").replace(end, ""))
        final_string = start_repl + str(mult * number) + end_repl
        file_content = file_content.replace(to_replace, final_string)

    return file_content
