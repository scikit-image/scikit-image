"""
Draft implementation of the Moore Neighborhood from
https://en.wikipedia.org/wiki/Moore_neighborhood
"""
import numpy as np

MOORE_OFFSETS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def find_moore_start(array):
    max_x = array.shape[0] - 1
    max_y = array.shape[1] - 1
    backtrack = [max_x, max_y]
    for x in range(max_x, -1, -1):
        for y in range(max_y, -1, -1):
            if array[x, y] == 1:
                return np.array(backtrack), np.array([x, y])
            backtrack = [x, y]
    return None, None


def next_moore_neighbor(current, backtrack, max_x, max_y):
    current_offset = (backtrack[0] - current[0], backtrack[1] - current[1])
    current_offset_index = MOORE_OFFSETS.index(current_offset)
    next_offsets = (
        MOORE_OFFSETS[current_offset_index + 1 :] + MOORE_OFFSETS[:current_offset_index]
    )
    for offset in next_offsets:
        if current[0] + offset[0] <= max_x and current[1] + offset[1] <= max_y:
            return current + offset


def moore(array):
    result = []
    max_x = array.shape[0] - 1
    max_y = array.shape[1] - 1
    backtrack, start = find_moore_start(array)
    result.append(start)
    current = next_moore_neighbor(start, backtrack, max_x, max_y)
    last_border = start
    while current[0] != start[0] or current[1] != start[1]:
        if array[current[0], current[1]] == 1:
            result.append(current)
            backtrack = last_border
            last_border = current
        else:
            backtrack = current
        current = next_moore_neighbor(last_border, backtrack, max_x, max_y)

    return result


example = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0],])
print(moore(example))

example_2 = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0],
    ]
)
print(moore(example_2))
