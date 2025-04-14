
import math
from math import factorial
import numpy as np


def display(points, base_points):
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_aspect('equal', adjustable='box')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.scatter(base_points[:, 0], base_points[:, 1],
               base_points[:, 2], linewidths=3)
    plt.show()


def display_xy(points, base_points):
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plt.xlim(0, 5)
    # plt.ylim(0, 5)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_aspect('equal', adjustable='box')

    ax.view_init(elev=-90, azim=90, roll=0)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    ax.scatter(base_points[:, 0], base_points[:, 1],
               base_points[:, 2], linewidths=3)
    plt.show()


def display_xz(points, base_points):
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(elev=0, azim=-90, roll=0)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.scatter(base_points[:, 0], base_points[:, 1],
               base_points[:, 2], linewidths=3)

    plt.show()


def display_yz(points, base_points):
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.view_init(elev=0, azim=0, roll=0)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.scatter(base_points[:, 0], base_points[:, 1],
               base_points[:, 2], linewidths=3)
    plt.show()


def bezie_surface(points: np.ndarray) -> np.ndarray:

    d = 10 ** -2 * 2

    m = points.shape[0]
    n = points.shape[1]
    surface = []


    for u in np.arange(0, 1 + d, d):
        for v in np.arange(0, 1 + d, d):
            acc = np.array([0., 0., 0., 1.])
            for i in range(m):
                Bmi = factorial(m) / (factorial(i) * factorial(m - i)) * u ** i * (1 - u) ** (m - i)
                for j in range(n):
                    Bnj = factorial(n) / (factorial(j) * factorial(n - j)) * v ** j * (1 - v) ** (n - j)
                    acc += Bmi * Bnj * points[i][j]
            surface.append(acc)

    surface = np.array(surface)

    return surface


def around_z(f):
    return np.array([[np.cos(f), -np.sin(f), 0, 0], [np.sin(f), np.cos(f), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def main():

    r = 1  # радиус шара
    c = r * 4 / 3 * (math.sqrt(2) - 1)  # вес для приближения дуги Безье

    points = [[[10, 0, 0], [10, 20, 30]],
              [[20, 0, 0], [30, 20, 30]]]

    # points = [ [ [0, 0, 0], [1, 0, 0]], [ [1, 1, 0], [0, 1, 0]] ]

    points = np.array([[list(p) + [1] for p in l] for l in points])

    bezie = bezie_surface(np.array(points))

    tr = around_z(np.pi / 4)
    sec = (tr @ bezie.T).T

    arr = []
    for l in points:
        for p in l:
            arr.append(p)

    points = np.array(arr)

    display(bezie, points)
    display_xy(bezie, points)
    display_xz(bezie, points)
    display_yz(bezie, points)

    # display_xy(sec, points)
    pass


if __name__ == '__main__':
    main()
