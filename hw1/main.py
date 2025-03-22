import math
from itertools import repeat

import numpy as np

import matplotlib.pyplot as plt

from numpy.linalg import norm
from numpy.linalg import inv


def display(polygons: np.ndarray):

    plt.figure()
    colors = np.array(['black', 'red', 'orange', 'yellow',
                      'green', 'blue', 'violet'])

    plt.axes()
    max_x = np.abs(polygons[:, :, 0]).max()
    max_y = np.abs(polygons[:, :, 1]).max()

    plt.xlim(-max_x - 0.5, max_x + 0.5)
    plt.ylim(- max_y - 0.5, max_y + 0.5)

    plt.axhline(linewidth=4, color='r')
    plt.axvline(linewidth=4, color='r')

    n_figs = polygons.shape[0]
    n_vertices = polygons.shape[1]
    plt.grid()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    fin_colors = []
    for color in colors[:n_figs]:
        fin_colors += list(repeat(color, n_vertices))

    for i, poly in enumerate(polygons):
        t = plt.Polygon(poly[:, (0, 1)], alpha=0.5,
                        color=colors[i % len(colors)])
        plt.gca().add_patch(t)

    plt.show()


def task_1():
    tri = np.array([[0, 0, 1], [0, 1, 1], [2, 0, 1]])
    a1 = (tr.perenos() @ tri.T).T

    display(np.array([tri, a1]))

    b1 = (tr.rotate(fi=np.pi / 3,
          axis=(tri[0] + tri[1] + tri[2]) / 3) @ tri.T).T

    display(np.array([tri, b1]))

    c1 = (tr.reflect(axis=np.array(
        [[-1, 0], [0, 1]]))  @ tr.homotetia() @ tri.T).T
    display(np.array([tri, c1]))

    sides_with_l = [((tri[0] + tri[1]) / 2, norm(tri[1] - tri[0])),
                    ((tri[0] + tri[2]) / 2, norm(tri[0] - tri[2])),
                    ((tri[1] + tri[2]) / 2, norm(tri[1] - tri[2]))]

    min_side_center = min(sides_with_l, key=lambda t: t[1])[0]

    d1 = (tr.homotetia(axis=min_side_center) @ tri.T).T
    e1 = (tr.rotate(fi=np.pi, axis=min_side_center) @
          tr.homotetia(axis=min_side_center) @ tri.T).T
    display(np.array([tri, d1, e1]))


def task_2():
    k = 3
    sqr = np.array([[1, 1, 1], [1, 1 + k, 1], [1 + k, 1 + k, 1],
                   [1 + k, 1, 1]], dtype=np.float64)

    # sqr = (tr.rotate(fi=np.pi/3) @ sqr.T).T

    a = norm(sqr[0] - sqr[1])
    angle = tr.angle_btwn_vecs(sqr[3] - sqr[0], [1, 0, 0])

    trans = tr.perenos(Tx=sqr[0][0], Ty=sqr[0][1])\
        @ tr.rotate(angle)\
        @ tr.perenos(Tx=a * 3, Ty=a * 3)\
        @ tr.shear(alpha=np.pi / 6, beta=0) \
        @ tr.homotetia(k=2)\
        @ tr.rotate(-angle) \
        @ tr.perenos(Tx=-sqr[0][0], Ty=-sqr[0][1])\


    para = (trans @ sqr.T).T

    para_inv = (inv(trans) @ para.T).T

    assert np.isclose(norm(sqr[0] - para[0]),  norm(sqr[0] - sqr[2]) * 3)

    assert np.isclose(norm(para[3] - para[0]), norm(sqr[0] - sqr[3]) * 2)

    assert np.isclose(tr.angle_btwn_vecs(para[1] - para[0], para[3] - para[0]),
                      np.pi / 3)

    assert np.all(np.isclose(para_inv, sqr, atol=tr.EPS))

    r_inv = (tr.homotetia(axis=para_inv[0]) @ para_inv.T).T
    display(np.array([sqr, para]))


def main():

    task_1()
    task_2()

    return 0


if __name__ == "__main__":
    import os
    file_path = os.path.realpath(__file__)

    shared_path = os.path.join(os.path.dirname(file_path), '..', 'shared')
    import sys
    sys.path.insert(0, shared_path)


    import display as dis
    import transforms as tr
    main()
