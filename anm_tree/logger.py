from treec.logger import GeneralLogger
import os
from treec.visualise_tree import binarytree_to_dot
import shutil
import timeit


class AnmLogger(GeneralLogger):
    def __init__(self, save_dir, algo_type, common_params, algo_params, tree_titles):
        super().__init__(save_dir, algo_type, common_params, algo_params)
        self.tree_titles = tree_titles

        self.dot_files_dir = self.folder_name + "dot_trees/"
        os.mkdir(self.dot_files_dir)


    def initialise_dir(self):
        params_str = ""
        params_str += self.algo_type + "\n\n"
        params_str += "common params:" + "\n"
        for key, value in self.common_params.items():
            if callable(value):
                val_str = value.__name__
            else:
                val_str = str(value)
            params_str += key + "," + val_str + "\n"

        params_str += "\nalgo params:\n"
        for key, value in self.algo_params.items():
            if callable(value):
                val_str = value.__name__
            else:
                val_str = str(value)
            params_str += key + "," + val_str + "\n"

        param_filepath = self.folder_name + "params_run.csv"

        text_file = open(param_filepath, "w+")
        text_file.write(params_str)
        text_file.close()

    def create_dir(self):

        env_name = self.common_params["gym_env"]
        folder_start = env_name + "_" + self.algo_type + "_"

        subfolders = [name for name in os.listdir(self.save_dir)]
        highest_num = -1
        for folder in subfolders:
            if folder.startswith(folder_start):
                folder_num = int(folder.replace(folder_start, ""))
                if folder_num > highest_num:
                    highest_num = folder_num

        folder_name = self.save_dir + "/" + folder_start + str(highest_num + 1) + "/"

        os.mkdir(folder_name)
        return folder_name

    def episode_eval_log(self, model, eval_score):
        episode_count, _ = self.read_best_score_episode_count()
        self.update_episode_count()

        score_str = "{0:.1f}".format(eval_score)

        eval_exten = str(episode_count) + "_" + score_str

        time_step = str(episode_count * self.common_params["tot_steps_train"])

        elapsed_time = str(timeit.default_timer() - self.start_time)

        text_file = open(self.eval_score_path, "a+")
        text_file.write(time_step + "," + elapsed_time + "," + score_str + "\n")
        text_file.close()

        better = self.update_best_score(eval_score)

        if better:
            self.save_model(model, eval_exten)
            return True
        return False

    def save_model(self, individual, eval_exten):
        if individual is not None:
            model_path = self.model_dir + "model_" + eval_exten + ".txt"

            file = open(model_path, "w+")
            model_str = ",".join([str(i) for i in individual])
            file.write(model_str)

            file.close()

    def save_tree_dot(self, trees, all_nodes_visited, eval_score):
        episode_count, _ = self.read_best_score_episode_count()

        score_str = "{0:.1f}".format(eval_score)

        eval_exten = str(episode_count) + "_" + score_str

        for i, tree in enumerate(trees):
            leafs = all_nodes_visited[i]
            title = self.tree_titles[i]
            dot_str = binarytree_to_dot(tree, title, leafs)
            file = open(self.dot_files_dir + eval_exten + "_" + title + ".dot", "w+")
            file.write(dot_str)
            file.close()
