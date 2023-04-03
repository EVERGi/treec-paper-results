from treec.visualise_tree import binarytree_to_dot

import os
import shutil
import timeit


class GeneralLogger:
    def __init__(self, save_dir, algo_type, common_params, algo_params):
        self.save_dir = save_dir
        self.algo_type = algo_type
        self.common_params = common_params
        self.algo_params = algo_params

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        self.folder_name = self.create_dir()

        self.initialise_dir()

        self.model_dir = self.folder_name + "models/"
        os.mkdir(self.model_dir)

        self.power_prof_dir = self.folder_name + "power_profiles/"
        os.mkdir(self.power_prof_dir)

        self.rewards_dir = self.folder_name + "rewards/"
        os.mkdir(self.rewards_dir)

        self.attributes_dir = self.folder_name + "attributes/"
        os.mkdir(self.attributes_dir)

        self.eval_score_path = self.folder_name + "eval_score.csv"

        file = open(self.eval_score_path, "w+")
        file.write("timestep,elapsed_time,eval_score\n")
        file.close()

        self.episode_score_file = self.folder_name + "episode_score_file.csv"
        self.create_episode_score_file()

        self.start_time = timeit.default_timer()

    def create_dir(self):
        config_file = self.common_params["config_file"]

        microgrid_name = config_file.split("/")[-1].split(".")[0]
        folder_start = microgrid_name + "_" + self.algo_type + "_"

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

    def initialise_dir(self):
        config_file = self.common_params["config_file"]
        logged_config_file = self.folder_name + "microgrid_config.csv"

        shutil.copyfile(config_file, logged_config_file)

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

    def episode_eval_log(self, microgrid, model, eval_score):
        episode_count, _ = self.read_best_score_episode_count()
        self.update_episode_count()

        score_str = "{0:.1f}".format(eval_score)

        eval_exten = str(episode_count) + "_" + score_str

        time_step = str(episode_count * self.common_params["tot_steps_train"])

        elapsed_time = str(timeit.default_timer() - self.start_time)

        text_file = open(self.eval_score_path, "a+")
        text_file.write(time_step + "," + elapsed_time + "," + score_str + "\n")
        text_file.close()

        better_score = self.update_best_score(eval_score)
        if better_score:
            self.log_microgrid_info(microgrid, eval_exten)

            self.save_model(model, eval_exten)
            return eval_exten

        return None

    def save_model(self, model, eval_exten):
        pass

    def create_episode_score_file(self):
        file = open(self.episode_score_file, "w+")
        file.write("episode,0\nscore,\n")
        file.close()

    def update_episode_count(self):
        file = open(self.episode_score_file, "r+")
        file_content = file.read().split("\n")
        file.close()
        try:
            episode_count = int(file_content[0].replace("episode,", ""))
        except (ValueError, IndexError):
            print(file_content)
            return

        new_episode_count = episode_count + 1

        file_content[0] = f"episode,{new_episode_count}"

        file = open(self.episode_score_file, "w+")
        file.write("\n".join(file_content))
        file.close()

    def update_best_score(self, new_best_score):
        file = open(self.episode_score_file, "r+")
        file_content = file.read().split("\n")
        file.close()
        try:
            best_score = file_content[1].replace("score,", "")
        except IndexError:
            print(file_content)
            return False

        if best_score == "" or float(best_score) < new_best_score:

            file_content[1] = f"score,{new_best_score}"

            file = open(self.episode_score_file, "w+")
            file.write("\n".join(file_content))
            file.close()

            return True

        return False

    def read_best_score_episode_count(self):
        file = open(self.episode_score_file, "r+")
        file_content = file.read().split("\n")
        file.close()
        try:
            episode_count = int(file_content[0].replace("episode,", ""))
        except (ValueError, IndexError):
            episode_count = 0
            print(file_content)

        try:
            if file_content[1].replace("score,", "") == "":
                best_score = None
            else:
                best_score = float(file_content[1].replace("score,", ""))
        except (ValueError, IndexError):
            best_score = 100
            print(file_content)

        return episode_count, best_score




class TreeLogger(GeneralLogger):
    def __init__(self, save_dir, algo_type, common_params, algo_params):
        super().__init__(save_dir, algo_type, common_params, algo_params)

        self.dot_files_dir = self.folder_name + "dot_trees/"
        os.mkdir(self.dot_files_dir)

    def save_model(self, individual, eval_exten):
        if individual is not None:
            model_path = self.model_dir + "model_" + eval_exten + ".txt"

            file = open(model_path, "w+")
            model_str = ",".join([str(i) for i in individual])
            file.write(model_str)

            file.close()

    def save_tree_dot(self, trees_batt, trees_charg, all_nodes_visited, eval_exten):

        for i, tree_batt in enumerate(trees_batt):
            leafs_batt = [j[i] for j in all_nodes_visited[0]]
            title = "Battery_" + str(i)
            dot_str = binarytree_to_dot(tree_batt, title, leafs_batt)
            file = open(self.dot_files_dir + eval_exten + "_" + title + ".dot", "w+")
            file.write(dot_str)
            file.close()

        for i, tree_charg in enumerate(trees_charg):
            leafs_charg = [j[i] for j in all_nodes_visited[1]]
            title = "Charger_" + str(i)
            dot_str = binarytree_to_dot(tree_charg, title, leafs_charg)
            file = open(self.dot_files_dir + eval_exten + "_" + title + ".dot", "w+")
            file.write(dot_str)
            file.close()
