from .tree import (
    BinaryTreeFixed,
    BinaryTreeFixedCont,
    BinaryTreeFree,
    BinaryTreeFreeCont,
)
from .norm_func import denormalise_input

from graphviz import Source
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from collections import Counter


def binarytree_to_dot(binarytree, title="", leafs_visited=None):

    if leafs_visited is not None:
        leafs_count = Counter(leafs_visited)
    dot_format = "digraph BST {\n"
    dot_format += '    node [fontname="Arial" style=filled colorscheme=paired12];\n'

    node_description = ""
    parent_links = ""
    for i, node in enumerate(binarytree.node_stack):
        value = node.value
        feature = node.feature
        if feature != -1:
            min_val = binarytree.feature_names[feature][1][0]
            max_val = binarytree.feature_names[feature][1][1]
            feature_value = denormalise_input(value, min_val, max_val)
            feature_value = "{:.2f}".format(round(feature_value, 2))

            color = node.feature % 12 + 1
            feature_name = binarytree.feature_names[feature][0]
            label = feature_name + ":\n" + str(feature_value)
            node_description += (
                "    "
                + str(i)
                + ' [ label = "'
                + label
                + '" fillcolor='
                + str(color)
                + "];\n"
            )
        else:
            if binarytree.discrete_actions:
                action_name = binarytree.action_names[value]
            else:
                value = denormalise_input(value, binarytree.act_min, binarytree.act_max)
                action_name = "{:.2f}".format(round(value, 2))
            if leafs_visited is None:
                label = action_name
            else:
                cur_index = binarytree.node_stack.index(node)

                label = action_name + "\nVisit count: " + str(leafs_count[cur_index])
            node_description += (
                "    " + str(i) + ' [ label = "' + label + '" fillcolor=white];\n'
            )
        left_node = node.left_node
        right_node = node.right_node
        if left_node is not None:
            left_index = binarytree.node_stack.index(left_node)
            right_index = binarytree.node_stack.index(right_node)
            parent_links += (
                "    " + str(i) + "  -> " + str(left_index) + '[ label = "<"];\n'
            )
            parent_links += (
                "    " + str(i) + "  -> " + str(right_index) + '[ label = ">="];\n'
            )

    dot_format += node_description + "\n" + parent_links + "\n"
    dot_format += '    labelloc="t";\n'
    dot_format += '    label="' + title + '";\n'
    dot_format += "}"
    return dot_format


def display_dot_matplotlib(dot_string):
    plt.figure()
    temp_file = "temp"
    s = Source(dot_string, filename=temp_file, format="png")
    s.render()
    img = mpimg.imread(temp_file + ".png")
    plt.imshow(img)

    os.remove(temp_file)
    os.remove(temp_file + ".png")


def display_binarytree(binarytree, title="", leafs_visited=None):
    bt_dot = binarytree_to_dot(binarytree, title, leafs_visited)
    display_dot_matplotlib(bt_dot)


if __name__ == "__main__":

    a = BinaryTreeFixed(
        [0.4, 0.8, 0.5, 0.4, 0, 0.4, 0.8],
        [["Feat 0", 1], ["Feat 1", 2], ["Feat 2", 1]],
        [
            "Action 0",
            "Action 1",
            "Action 2",
        ],
    )
    b = BinaryTreeFree(
        [0.6, 0.8, 0.1, 0.1, 0.1, 0.5, 0.4, 0.4, 0.8, 0.1],
        [["Feat 0", 2], ["Feat 1", 1], ["Feat 2", 1]],
        [
            "Action 0",
            "Action 1",
            "Action 2",
        ],
    )

    a = BinaryTreeFixedCont(
        [0.4, 0.8, 0.5, 0.4, 0, 0.4, 0.8],
        [["Feat 0", 1], ["Feat 1", 2], ["Feat 2", 1]],
        [],
    )
    b = BinaryTreeFreeCont(
        [0.6, 0.8, 0.1, 0.1, 0.1, 0.5, 0.4, 0.4, 0.8, 0.1],
        [["Feat 0", 2], ["Feat 1", 1], ["Feat 2", 1]],
        [],
    )

    a_dot = binarytree_to_dot(a)
    b_dot = binarytree_to_dot(b)
    print(a_dot)
    print(b_dot)

    display_dot_matplotlib(a_dot)
    display_dot_matplotlib(b_dot)
    plt.show()
