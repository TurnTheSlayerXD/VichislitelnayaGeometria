

import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt


def display_points(points, segments, labels=['unknown']):
    colors = np.array(['black', 'red', 'orange', 'yellow',
                      'green', 'blue', 'violet'])
    plt.grid()
    plt.scatter(points[:, 0], points[:, 1], color=colors[0])
    plt.plot(segments[:, 0], segments[:, 1], color=colors[1])
    last = np.array([segments[-1], segments[0]])
    plt.plot(last[:, 0], last[:, 1], color=colors[1])
    np.clip
    plt.show()

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def find_start_point(points : np.ndarray):
    
    res = [sys.maxsize, sys.maxsize]
    ind = sys.maxsize
    for i, p in enumerate(points):
        if p[1] < res[1] or p[1] == res[1] and p[0] < res[0]:
            res = p
            ind = i 
    return ind

def sort_related(p0, points):
    base = np.array([p0[0] + 10, p0[1]]) - p0
    return sorted(points, key=lambda p: angle_between(base,p - p0))

def conv_grakham(points: np.ndarray):

    p0_ind = find_start_point(points)
    
    to_sort = np.concat([points[0 : p0_ind],
                        points[p0_ind+1:len(points)]])
    sorted_ps = sort_related(points[p0_ind], to_sort)
    p0 = points[p0_ind]

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

def gen_points(count, sqr_size):
    res = np.random.random_sample((count, 2)) * sqr_size

    return res

def main():
    warnings.filterwarnings('ignore')

    points = gen_points(10**2, 10**3)
    conv = conv_grakham(points)
    
    # print(points)

    # print(conv.shape)
    display_points(points, conv)

if __name__ == '__main__':
    main()
