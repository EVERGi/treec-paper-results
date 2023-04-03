from collections import Counter

def prune_unvisited_nodes(binarytree, leafs_visited_index):
    
    leafs_visited_node = [binarytree.node_stack[i] for i in leafs_visited_index]
    leafs_count = Counter(leafs_visited_node)
    
    if len(leafs_count) == 1:
        only_node = list(leafs_count.keys())[0]
        binarytree.root_node = only_node
        only_node.root = True
        binarytree.recalculate_node_stack()
        return [0 for _ in leafs_visited_index]
    left_node = binarytree.root_node.left_node
    right_node = binarytree.root_node.right_node

    remove_or_not_node(left_node, leafs_count, binarytree)
    remove_or_not_node(right_node, leafs_count, binarytree)

    new_index = {node: i for i, node in enumerate(binarytree.node_stack)}
    new_leafs_visited_index = [new_index[i] for i in leafs_visited_node]
    return new_leafs_visited_index


def remove_or_not_node(cur_node, leafs_count, binarytree):
    parent_node = binarytree.get_parent(cur_node)
    
    if cur_node.feature == -1:
        if leafs_count[cur_node] == 0:
            if parent_node.left_node is cur_node:
                node_to_keep = parent_node.right_node
            else:
                node_to_keep = parent_node.left_node

            if parent_node.root:
                binarytree.root_node = node_to_keep
                node_to_keep.root = True

            else:
                grand_parent_node = binarytree.get_parent(parent_node)
                if grand_parent_node.left_node is parent_node:
                    grand_parent_node.left_node = node_to_keep
                else:
                    grand_parent_node.right_node = node_to_keep
            binarytree.recalculate_node_stack()
    else:
        left_node = cur_node.left_node
        right_node = cur_node.right_node
        remove_or_not_node(left_node, leafs_count, binarytree)
        remove_or_not_node(right_node, leafs_count, binarytree)

def prune_same_action_nodes(binarytree, leafs_visited_index):
    leafs_visited_node = [binarytree.node_stack[i] for i in leafs_visited_index]
    
    binarytree.recalculate_depth()
    max_depth = max([node.depth for node in binarytree.node_stack])

    for depth in reversed(range(max_depth)):
        for node in binarytree.node_stack:
            
            if node.left_node is None or node.right_node is None or node.depth != depth:
                continue
            
            children_are_leafs = node.left_node.feature == -1 and node.right_node.feature == -1
            same_value = node.left_node.value == node.right_node.value

            if children_are_leafs and same_value:
                if node.root:
                    binarytree.root_node = node.left_node
                    node.left_node.root = True
                else:
                    parent =  binarytree.get_parent(node)
                    if parent.right_node is node:
                        parent.right_node = node.left_node
                    else:
                        parent.left_node = node.left_node
                
                leafs_visited_node = [node.left_node if i is node.right_node else i for i in leafs_visited_node]
                binarytree.recalculate_depth()

    new_index = {node: i for i, node in enumerate(binarytree.node_stack)}
    new_leafs_visited_index = [new_index[i] for i in leafs_visited_node]

    return new_leafs_visited_index


def prune_tree(binarytree, leafs_visited_index):

    leafs_visited_index = prune_unvisited_nodes(binarytree, leafs_visited_index)

    new_leafs_visited_index = prune_same_action_nodes(binarytree, leafs_visited_index)

    return new_leafs_visited_index
