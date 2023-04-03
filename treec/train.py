from .tree import (
    BinaryTreeFixed,
    BinaryTreeFree,
    BinaryTreeFixedCont,
    BinaryTreeFreeCont,
)
import os


def find_best_individual(folder):
    model_files = [
        f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))
    ]

    best_model = model_files[0]

    for model_file in model_files:
        best_score = float(best_model.split("_")[-1].replace(".txt", ""))
        model_score = float(model_file.split("_")[-1].replace(".txt", ""))

        if model_score > best_score:
            best_model = model_file
    best_model_file = folder + best_model

    file = open(best_model_file, "r")
    print(file)

    indiv_str_list = file.read().split(",")

    indiv = [float(i) for i in indiv_str_list]

    return indiv


def get_treestruct(common_params, algo_params):
    continuous = common_params["continuous"]
    fixed = algo_params["fixed"]

    if fixed and continuous:
        TreeStruct = BinaryTreeFixedCont
    elif fixed and not continuous:
        TreeStruct = BinaryTreeFixed
    elif not fixed and continuous:
        TreeStruct = BinaryTreeFreeCont
    else:
        TreeStruct = BinaryTreeFree

    return TreeStruct
