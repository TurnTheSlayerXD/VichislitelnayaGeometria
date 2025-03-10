import math
from itertools import repeat

import numpy as np

import matplotlib.pyplot as plt


def display(polygons: np.ndarray):
    
    plt.figure()
    colors = np.array(['black', 'red', 'orange', 'yellow', 'green', 'blue', 'violet'])

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

    for i, poly in enumerate(polygons):
        t = plt.Polygon(poly[:, (0, 1)], alpha=0.5, color=colors[i % len(colors)])
        plt.gca().add_patch(t)
        print(t)

    plt.show()


def task_1():
    tri = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1]])
    a1 = (tr.perenos() @ tri.T).T
    b1 = (tr.rotate(fi=np.pi / 3) @ tri.T).T
    c1 = (tr.reflect() @ tri.T).T
    d1 = (tr.homotetia() @ tri.T).T
    
    side_center = (tri[1] + tri[2]) / 2
    e1 = ((tr.homotetia(axis=side_center) @ tr.rotate(fi=np.pi)) @ tri.T).T
    f1 = (tr.homotetia(axis=side_center) @ tri.T).T
    
    display(np.array([tri, a1, b1, c1, d1, e1, f1]))


from numpy.linalg import inv


def task_2():
    sqr = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]])
    trans = tr.perenos(Tx=2) @ tr.shear(alpha=np.pi / 3, beta=np.pi / 3)
    r1 = (trans @ sqr.T).T
    r1_inv = (tr.rotate(fi=np.pi / 3) @ inv(trans) @ r1.T).T
    display(np.array([sqr, r1, r1_inv]))


def main():
    
    task_1()
    task_2()
    
    return 0


if __name__ == "__main__":
    import sys 
    sys.path.insert(0, 'C:/Users/Professional/Desktop/VichislitelnayaGeometria/shared')
    import display as dis 
    import transforms as tr 
    main()
