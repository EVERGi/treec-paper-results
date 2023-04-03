import ast
from typing import List
import abc


class NodeBinary:
    def __init__(self, feature: int, value: float, depth: int):

        self.feature = feature
        self.value = value
        self.depth = depth

        self.left_node = None
        self.right_node = None

        self.root = False


class BinaryTree:
    def __init__(
        self, node_array: List[float], feature_names: List[str], action_names: List[str]
    ):
        self.feature_names = feature_names
        self.action_names = action_names

        self.num_features = len(feature_names)
        self.num_actions = len(action_names)

        self.node_stack = list()
        self.root_node = None

        if self.num_actions != 0:
            self.discrete_actions = True
        else:
            self.discrete_actions = False

        self.act_min = 0
        self.act_max = 1

        self.construct_tree(node_array)

    def set_act_min_max(self, act_min, act_max):
        self.act_min = act_min
        self.act_max = act_max

    @abc.abstractmethod
    def construct_tree(self, node_array: List[float]):
        pass

    def get_action(self, feature_values: List[float]):
        cur_node = self.root_node

        while cur_node.feature != -1:
            val = feature_values[cur_node.feature]
            if val < cur_node.value:
                cur_node = cur_node.left_node
            else:
                cur_node = cur_node.right_node
        return cur_node

    def __str__(self):
        tree_struct = ""
        for node in self.node_stack:
            if node.feature == -1:
                feature = ""
            else:
                feature = str(node.feature) + ":"
            tree_struct += node.depth * " " + feature + str(node.value) + "\n"

        return tree_struct

    def recalculate_node_stack(self):
        new_stack = list()
        sort_stack(new_stack, self.root_node)
        self.node_stack = new_stack

    def recalculate_depth(self):
        self.recalculate_node_stack()
        for node in self.node_stack:
            if node.root:
                node.depth = 0
            else:
                parent = self.get_parent(node)
                node.depth = parent.depth + 1

    def get_parent(self, child):
        for node in self.node_stack:
            if node.left_node is child or node.right_node is child:
                return node
        return None

    def get_free_node_array(self):
        features = list()
        values = list()
        self.recalculate_node_stack()

        for node in self.node_stack:
            features.append((node.feature + 1) / (self.num_features + 1))
            if self.discrete_actions:
                values.append(node.value / self.num_actions)
            else:
                values.append(node.value)

        node_array = features + values

        return node_array

    def save_tree_as_free(self, filepath):

        node_array = []
        node_array = self.get_free_node_array(self.root_node, node_array)

        file = open(filepath, "w+")
        file.write(f"{node_array}\n")
        file.write(f"{self.feature_names}\n")
        file.write(f"{self.action_names}\n")
        file.close()

    def get_free_node_array(self, node, node_array):

        feature_double = (node.feature + 1) / (self.num_features + 1)
        mid_array = int(len(node_array) / 2)
        node_array = node_array[:mid_array] + [feature_double] + node_array[mid_array:]

        if node.feature == -1 and self.discrete_actions:
            value_double = node.value / self.num_actions
        else:
            value_double = node.value

        node_array += [value_double]

        if node.left_node is not None:
            node_array = self.get_free_node_array(node.left_node, node_array)
        if node.right_node is not None:
            node_array = self.get_free_node_array(node.right_node, node_array)
        return node_array


def construc_tree_from_file(filepath):
    file = open(filepath, "r+")
    content = file.readlines()
    print(content)

    node_array = ast.literal_eval(content[0][:-1])
    feature_names = ast.literal_eval(content[1][:-1])
    action_names = ast.literal_eval(content[2][:-1])

    return BinaryTreeFree(node_array, feature_names, action_names)


class BinaryTreeFree(BinaryTree):
    def __init__(
        self, node_array: List[float], feature_names: List[str], action_names: List[str]
    ):
        super().__init__(node_array, feature_names, action_names)

    def construct_tree(self, node_array: List[float]):
        split_features = [
            int(i * (self.num_features + 1) - 1)
            for i in node_array[: int(len(node_array) / 2)]
        ]
        split_values = node_array[int(len(node_array) / 2) :]

        uncompleted_stack = list()
        leaf_to_complete = 0

        depth = 0

        for i, feature in enumerate(split_features):

            nodes_left = len(split_features) - i
            if leaf_to_complete >= nodes_left - 1:
                feature = -1
            if feature == -1:
                if self.discrete_actions:
                    current_value = int(split_values[i] * self.num_actions)
                else:
                    current_value = split_values[i]
            else:
                current_value = split_values[i]
            current_node = NodeBinary(feature, current_value, depth)
            self.node_stack.append(current_node)

            if i == 0:
                self.root_node = current_node
                current_node.root = True

                if feature == -1:
                    break

                uncompleted_stack.append(current_node)
                leaf_to_complete += 2
            else:
                if uncompleted_stack[-1].left_node is None:
                    uncompleted_stack[-1].left_node = current_node
                elif uncompleted_stack[-1].right_node is None:
                    uncompleted_stack[-1].right_node = current_node
                    uncompleted_stack.pop()

                if feature == -1:
                    leaf_to_complete -= 1
                else:
                    leaf_to_complete += 1
                    uncompleted_stack.append(current_node)

            if leaf_to_complete == 0 and len(uncompleted_stack) == 0:
                break

            depth = uncompleted_stack[-1].depth + 1


class BinaryTreeFixed(BinaryTree):
    def __init__(
        self, node_array: List[float], feature_names: List[str], action_names: List[str]
    ):
        if (len(node_array) - 1) % 3 != 0:
            message = """The list should have a length of 3n+1 with n a positive integer 
            where the first n elements are the features of the tree, 
            the next n are the split values 
            and the next n+1 the values of the leafs"""
            raise (ValueError(message))

        self.num_splits = int((len(node_array) - 1) / 3)

        super().__init__(node_array, feature_names, action_names)

        self.recalculate_node_stack()

    def construct_tree(self, node_array: List[float]):
        split_features = [
            int(i * self.num_features) for i in node_array[: self.num_splits]
        ]
        split_values = node_array[self.num_splits : 2 * self.num_splits]
        if self.discrete_actions:
            leaf_values = [
                int(i * self.num_actions) for i in node_array[2 * self.num_splits :]
            ]
        else:
            leaf_values = [i for i in node_array[2 * self.num_splits :]]
        uncompleted_stack = list()

        depth = 0

        for i, feature in enumerate(split_features):

            current_value = split_values[i]
            current_node = NodeBinary(feature, current_value, depth)
            self.node_stack.append(current_node)

            if i == 0:
                self.root_node = current_node
                current_node.root = True

            elif uncompleted_stack[0].left_node is None:
                uncompleted_stack[0].left_node = current_node
            elif uncompleted_stack[0].right_node is None:
                uncompleted_stack[0].right_node = current_node
                uncompleted_stack.pop(0)

            uncompleted_stack.append(current_node)
            depth = uncompleted_stack[0].depth + 1

        for i, leaf_value in enumerate(leaf_values):
            current_node = NodeBinary(-1, leaf_value, depth)
            self.node_stack.append(current_node)

            if len(leaf_values) == 1:
                self.root_node = current_node
                break
            elif uncompleted_stack[0].left_node is None:
                uncompleted_stack[0].left_node = current_node
            elif uncompleted_stack[0].right_node is None:
                uncompleted_stack[0].right_node = current_node
                uncompleted_stack.pop(0)
            if len(uncompleted_stack) != 0:
                depth = uncompleted_stack[0].depth + 1


class BinaryTreeFreeCont(BinaryTreeFree):
    def __init__(
        self, node_array: List[float], feature_names: List[str], action_names: List[str]
    ):
        super().__init__(node_array, feature_names, [])


class BinaryTreeFixedCont(BinaryTreeFixed):
    def __init__(
        self, node_array: List[float], feature_names: List[str], action_names: List[str]
    ):
        super().__init__(node_array, feature_names, [])


def sort_stack(new_stack, cur_node):
    new_stack.append(cur_node)
    if cur_node.feature == -1:
        return
    else:
        sort_stack(new_stack, cur_node.left_node)
        sort_stack(new_stack, cur_node.right_node)


if __name__ == "__main__":
    a = BinaryTreeFixed(
        [0.4, 0.8, 0.5, 0.4, 0, 0.4, 0.8],
        ["Feat 0", "Feat 1", "Feat 2"],
        ["Action 0", "Action 1", "Action 2"],
    )
    b = BinaryTreeFree(
        [0.6, 0.8, 0.1, 0.1, 0.1, 0.5, 0.4, 0.4, 0.8, 0.1],
        ["Feat 0", "Feat 1", "Feat 2"],
        ["Action 0", "Action 1", "Action 2"],
    )

    # a = BinaryTreeFixedCont([0.4,0.8,0.5,0.4,0.1,0.4,0.8], ["Feat 0", "Feat 1", "Feat 2"])
    # b = BinaryTreeFreeCont([0.6,0.8,0.1,0.1,0.1,0.5,0.4,0.4,0.8,0.1], ["Feat 0", "Feat 1", "Feat 2"])

    print(a)
    print(b)

    print(a.get_action([0.2, 0.6, 0.6]))
    print(b.get_action([0.2, 0.3, 0.3]))
