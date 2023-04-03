from .tree import BinaryTree

from typing import List, Callable


def normalise_input(input, min_val, max_val):
    pow_range = max_val - min_val

    # If the asset can either charge or discharge, scale only on that axis
    # If asset can charge and discharge, then 0.5 does nothing
    if pow_range == 0:
        output = 0
    elif min_val * max_val >= 0:
        output = (input - min_val) / pow_range

    elif input > 0:
        output = input / max_val / 2 + 0.5
    else:
        output = 0.5 - input / min_val / 2

    return output


def denormalise_input(input, min_val, max_val):
    pow_range = max_val - min_val
    # If the asset can either charge or discharge, scale only on that axis
    # If asset can charge and discharge, then 0.5 does nothing
    if min_val * max_val >= 0:
        output = input * pow_range + min_val
    elif input > 0.5:
        output = 2 * (input - 0.5) * max_val
    else:
        output = 2 * (input - 0.5) * (-min_val)
    return output
