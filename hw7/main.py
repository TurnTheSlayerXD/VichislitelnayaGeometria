

from functools import reduce

from bisect import bisect_left

import numpy as np
import matplotlib.pyplot as plt


def dist(p1, p2) -> np.float64:
    return np.float64(np.linalg.norm(p2 - p1))


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


def recurs(points: np.ndarray, l, r):

    assert r - l > 1

    if (r - l <= 3):
        if (r - l == 2):
            return points[l], points[l+1]
        else:
            dists = [(points[l], points[l+1]),
                     (points[l], points[l+2]),
                     (points[l+1], points[l+2])]
            dists.sort(key=lambda ps: dist(ps[0], ps[1]))
            return dists[0][0], dists[0][1]

    mid = (l + r) // 2
    l1, l2 = recurs(points, l, mid)
    r1, r2 = recurs(points, mid, r)

    (min_dist, min_p1, min_p2) = (dist(l1, l2), l1, l2) if dist(l1, l2) < dist(r1, r2)\
        else (dist(r1, r2), r1, r2)
    x_mean = points[mid][0]

    B_left = points[l:bisect_left(
        points, x_mean - min_dist, l, mid, key=lambda p: p[0])]
    B_right = points[mid:bisect_left(
        points, x_mean + min_dist, mid, r, key=lambda p: p[0])]

    B = B_left.tolist() + B_right.tolist()

    B.sort(key=lambda p: p[1])

    for i in range(len(B)):
        for j in range(i - 1, -1, -1):

            val = dist(np.array(B[i]), np.array(B[j]))
            if val < min_dist:
                min_dist = val
                min_p1 = B[i]
                min_p2 = B[j]

    return min_p1, min_p2


def closest_points(points):
    points = np.array(list(sorted(points, key=lambda x: x[0])))
    return recurs(points, 0, len(points))


def main():

    points = np.array([[3.5, 0], [2.5, 4], [4, 3],
                       [5, 1.5], [3, 0], [2, 2], [1, 1]])

    p1, p2 = closest_points(points)

    display_points(points, np.array([p1, p2]))

    print(p1, p2)


if __name__ == '__main__':

    main()
