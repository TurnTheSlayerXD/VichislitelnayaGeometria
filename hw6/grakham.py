
import numpy as np
import sys


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return angle


def find_start_point(points: np.ndarray):

    res = [sys.maxsize, sys.maxsize]
    ind = sys.maxsize
    for i, p in enumerate(points):
        if p[1] < res[1] or p[1] == res[1] and p[0] < res[0]:
            res = p
            ind = i
    return ind


def sort_related(p0, points):
    base = np.array([1, 0])
    return sorted(points, key=lambda p: angle_between(base, p - p0))


def grakham_convex(points: np.ndarray) -> np.ndarray:

    p0_ind = find_start_point(points)

    to_sort = np.concat([points[0: p0_ind],
                        points[p0_ind+1:len(points)]])
    sorted_ps = sort_related(points[p0_ind], to_sort)
    p0 = points[p0_ind]

    if len(sorted_ps) == 0:
        return np.array(p0)
        
    stack = [p0, sorted_ps[0]]
    for i in range(1, len(sorted_ps)):
        assert len(stack) >= 2
        next_to_top = stack[-2]
        top = stack[-1]
        pi = sorted_ps[i]
        while np.cross(pi - top, next_to_top - top) < 0:
            stack.pop()
            
            assert len(stack) >= 2
            next_to_top = stack[-2]
            top = stack[-1]
        stack.append(pi)
        
    return np.array(stack)
