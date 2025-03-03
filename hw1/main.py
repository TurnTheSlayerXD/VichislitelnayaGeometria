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

    plt.grid()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    fin_colors = [] 
    for color in colors[:n_figs]:
        fin_colors += list(repeat(color, n_vertices))

    # plt.scatter(cords[:, 0], cords[:, 1], s=170,
    #             color=fin_colors)
    for i, poly in enumerate(polygons):
        t = plt.Polygon(poly[:, (0, 1)], alpha=0.5, color=colors[i])
        plt.gca().add_patch(t)
        print(t)

    plt.show()


def perenos(arr, Tx=2, Ty=0):
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


def task_1_a():
    tri = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1]])
    res = perenos(tri)
    display(np.array([tri, res]))


def task_1_b():
    tri = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1]])
    
    res = rotate(tri, fi=np.pi / 3)
    display(np.array([tri, res]))


def task_1_c():
    tri = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1]])
    
    res = reflect(tri)
    display(np.array([tri, res]))


def task_1_d():
    tri = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1]])
    
    res = homotetia(tri)
    display(np.array([tri, res]))

    
def task_1_e(side_num=0):
    tri = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1]])
    side_center = (tri[side_num % 3] + tri[ (side_num + 1) % 3]) / 2
    
    r1 = homotetia(tri, axis=side_center)
    r2 = rotate(r1, axis=side_center)

    display(np.array([tri, r1, r2]))


def task_2_a():
    pass


def task_2_b():
    sqr = np.array([[1, 1, 1], [1, 2, 1], [2, 2, 1], [2, 1, 1]])
    r1 = shear(sqr)
    r2 = shear(sqr, alpha=0)
    r3 = shear(sqr, beta=0)
    display(np.array([sqr, r1, r2, r3]))
    pass


def task_2_c():
    pass


def main():
    
    task_1_e()
    
    return 0


if __name__ == "__main__":
    main()
