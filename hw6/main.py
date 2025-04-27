

from functools import reduce

from bisect import bisect_left, bisect_right

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

    by_x = lambda p: p[0]
    by_y = lambda p: p[1]
    assert r - l > 1

    if (r - l <= 3):
        if (r - l == 2):
            return points[l], points[l+1]
        else:
            dists = [(points[l], points[l+1]),
                     (points[l], points[l+2]),
                     (points[l+1], points[l+2])]
            dists.sort(key=lambda seg: dist(seg[0], seg[1]))
            return dists[0][0], dists[0][1]

    mid = (l + r) // 2
    
    
    l1, l2 = recurs(points, l, mid)
    r1, r2 = recurs(points, mid, r)

    assert type(r1) is np.ndarray and type(r2) is np.ndarray
    
    assert all([points[i][0] < points[i+1][0] for i in range(len(points) - 1)])
    
    dist_left = dist(l1, l2)
    dist_right = dist(r1, r2)
    
    (min_dist, min_p1, min_p2) = (dist_left, l1, l2) if dist_left < dist_right\
        else (dist_right, r1, r2)
        
    x_mean = points[mid][0]

    B_left = points[bisect_left(points, x_mean - min_dist, l , mid, key=by_x):mid]
    
    B_right = points[mid:bisect_right(points, x_mean + min_dist, mid, r, key=by_x)]
    print('\n')

    print(f'points={points[l:r]}')
    print(f'left={B_left}')
    print(f'right={B_right}')
    

    B_right = np.array(sorted(B_right, key=by_y))

    assert all([B_right[i][1] < B_right[i+1][1] for i in range(len(B_right) - 1)])

    distance = np.copy(min_dist)
    
    for p_left in B_left:
        low = bisect_left(B_right, p_left[1] - distance, key=by_y)
        high = bisect_right(B_right, p_left[1] + distance, key=by_y)
        for p_right in B_right[low - 1:high + 1]:
            val = dist(p_left, p_right)
            if val < min_dist:
                min_dist = val
                print(min_dist)
                min_p1 = p_left
                min_p2 = p_right

    return np.array(min_p1), np.array(min_p2)


def closest_points(points):
    points = np.array(list(sorted(points, key=lambda x: x[0])))
    return recurs(points, 0, len(points))



def main():

    points = np.array([[3.5, 0], [2.5, 4], [4, 3],
                       [5, 1.5], [3, 0], [2, 2], [1, 1]])

    points = np.random.rand(20, 2)
    
    
                            

    print(f'Points = {points}')
    p1, p2 = closest_points(points)

    display_points(points, np.array([p1, p2]))

    print(p1, p2)



if __name__ == '__main__':

    main()


