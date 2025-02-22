import math
from itertools import repeat

import numpy as np

import matplotlib.pyplot as plt


def display(polygons: np.ndarray):
    
    plt.figure()
    colors = np.array(['red', 'green', 'blue', 'yellow'])

    max_x = np.abs(polygons[:,:, 0]).max()
    max_y = np.abs(polygons[:,:, 1]).max()

    fin = np.max([max_x, max_y])
    
    plt.xlim(-fin, fin)
    plt.ylim(-fin, fin)

    n_figs = polygons.shape[0]
    n_vertices = polygons.shape[1]

    assert n_vertices == 4

    plt.grid()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    fin_colors = [] 
    for color in colors[:n_figs]:
        fin_colors += list(repeat(color, n_vertices))

    # plt.scatter(cords[:, 0], cords[:, 1], s=170,
    #             color=fin_colors)
    from matplotlib.patches import RegularPolygon
    for i, poly in enumerate(polygons):
        t = plt.Polygon(poly[:, (0, 1)], alpha=0.5, color=colors[i])
        plt.gca().add_patch(t)

    plt.show()


def perenos(arr, Tx=10, Ty=0):
    mat = np.array([[1, 0, Tx], [0, 1, Ty], [0, 0, 1]])
    res = np.dot(mat, arr.transpose()).transpose()
    return res


def rotate(arr, fi=math.pi / 2, axis=np.array([0, 0])):
    t1 = perenos(arr, -axis[0], -axis[1])
    
    mat = np.array([[np.cos(fi), np.sin(fi), 0], [-np.sin(fi), np.cos(fi), 0], [0, 0, 1]])
    
    t2 = (mat @ t1.transpose()).transpose()

    t3 = perenos(t2, axis[0], axis[1]) 

    return t3


def reflect(arr, axis='x'):
    mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    if axis == 'x':
        mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    return np.dot(mat, arr.transpose()).transpose()


def homotetia(arr, axis=np.array([0, 0]), k=2):
    mat = np.dot(k, np.array([ [1, 0, -axis[0]], [0, 1, -axis[1]], [0, 0, 1 / k] ]))
    res = np.dot(mat, arr.transpose()).transpose()    
    return perenos(res, axis[0], axis[1])


def composition(arr):
    f = rotate(arr, math.pi)
    s = homotetia(f)
    return s


def shear(arr: np.ndarray, alpha=math.pi / 3, beta=math.pi / 3):

    mat = np.array([[1, np.tan(alpha), 0], [np.tan(beta), 1, 0], [0 , 0 , 1]])

    res = (mat @ arr.transpose()).transpose()

    return res


def task_1_e(arr: np.ndarray, side_num=0):
    side_center = (arr[side_num % 3] + arr[ (side_num + 1) % 3]) / 2
    
    r1 = homotetia(arr, axis=side_center)
    r2 = rotate(r1, axis=side_center)

    return r1, r2


def task_2():
    arr = [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]]
    
    pass


def main():
    triangle = np.array([[1, 1, 1], [1, 2, 1], [2, 2, 1], [2, 1, 1]])
    r1 = shear(triangle)
    r2 = shear(triangle, alpha=0)
    r3 = shear(triangle, beta=0)


    display(np.array([triangle, r1, r2, r3]))

    return 0


if __name__ == "__main__":
    main()
