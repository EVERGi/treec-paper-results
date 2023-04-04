import argparse
from boptest.train import train_function_boptest
from anm6easy.train import train_function_anm6easy


def boptest_train(scenario, train_gen):
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
    
    folder_name = train_function_boptest("5000", electricity_price, time_period, train_gen)
    return folder_name

def anm6easy_train(scenario, train_gen):
    folder_name = train_function_anm6easy(scenario, train_gen)

    return folder_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute the trainings of TreeC of visualise the trees obtained for the results in the paper.')


    parser.add_argument('case', type=str, choices=["anm", "bop"],
                        help='Select anm6easy (a) or botpest case (b).',required=True)
    parser.add_argument('mode', type=str, choices=["train", "visu"],
                        help='Choose train or visualisation mode.',required=True)
    parser.add_argument('scenario', type=int,
                        help='Which scenario of the study cases to use, seed for anm6easy or int from 0 to 5 for the 6 different boptest scenarios.',required=True)
    parser.add_argument('tree_path', type=str,
                        help='Path to tree with requested visualisation. (default: Tree in paper)',required=False)
    parser.add_argument('train_gen', type=int,
                        help='Number of generations for the training. (default: Number of generations in paper)',required=False)
    
    args = parser.parse_args()

    if args.mode == "train":
        if args.case == "anm":
            anm6easy_train(args.scenario, args.train_gen)
        elif args.case == "bop":
            boptest_train(args.scenario, args.train_gen)