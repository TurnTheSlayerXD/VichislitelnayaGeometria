import math
from itertools import repeat

import numpy as np

import matplotlib.pyplot as plt


def display(triangles: np.ndarray):
    count = triangles.shape[0] * triangles.shape[1]
    cords = triangles.reshape([triangles.shape[0] * triangles.shape[1], 3])
    plt.figure()
    colors = np.array(['red', 'green', 'blue', 'yellow'])

    fin_colors = []
    for color in colors[:triangles.shape[0]]:
        fin_colors += list(repeat(color, triangles.shape[1]))
    print(fin_colors)
    plt.scatter(cords[:, 0], cords[:, 1], s=170,
                color=fin_colors)
    for i, tr in enumerate(triangles):
        t = plt.Polygon(tr[:, (0, 1)], color=colors[i])
        plt.gca().add_patch(t)

    plt.show()


def perenos(arr, Tx=10, Ty=0):
    mat = np.array([[1, 0, Tx], [0, 1, Ty], [0, 0, 1]])
    res = np.dot(mat, arr.transpose()).transpose()
    return res


def rotate(arr, fi=math.pi / 2):
    mat = np.array([[np.cos(fi), np.sin(fi), 0], [-np.sin(fi), np.cos(fi), 0], [0, 0, 1]])
    return np.dot(mat, arr.transpose()).transpose()


def reflect(arr, axis='x'):
    mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    if axis == 'x':
        mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    return np.dot(mat, arr.transpose()).transpose()


def homotetia(arr):
    return arr



def composition(arr):
    f = rotate(arr, math.pi)
    s = homotetia(f)

    return s
def task_2():
    arr = [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]]


    pass

def main():
    triangle = np.array([[0, 0, 1], [0, 1, 1], [3, 0, 1]])

    conv_triangle = rotate(triangle)

    print(triangle)
    print(conv_triangle)
    display(np.array([triangle, conv_triangle]))

    return 0


if name == "__main__":
    main()