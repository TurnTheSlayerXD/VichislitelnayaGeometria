

import numpy as np

import matplotlib.pyplot as plt



import matplotlib.pyplot as plt

def task_1(A=np.array([-10, -7, 1]), B=np.array([15, 20, 1])):
    x0, y0, _ = A
    x1, y1, _ = B

    steep = y1 - y0 > x1 - x0
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    
    dx = np.abs(x1 - x0)
    dy = np.abs(y1 - y0)
    
    error = dx / 2
    ystep = 1 if y0 < y1 else -1 
    
    y = y0
    points = []
    for x in range(x0, x1 + 1):
        points.append([y, x] if steep else [x, y])
        error -= dy
        if error < 0:
            y += ystep
            error += dx

    k = dy / dx
    c = B[1] - k * B[0] 

    x_ = list(range(A[0], B[0] + 1))
    points2 = np.array([ [x, int(k * x + c)] for x in x_])


    points = np.array(points)

    plt.grid()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.plot(points[:, 0], points[:, 1], 'o', color='black', markersize=4,
            label='Brezenheim')
    plt.plot(points2[:, 0], points2[:, 1], 'o', color='red', markersize=2,
             label='By equation')
    plt.legend(loc='best')
    plt.show()



def task_2_a():
    points = np.array([[-1, -1, 1], [-1, 1, 1], [1, 1, 1],
                      [1, -1, 1]], dtype=np.float64)

    final = tr.bezie_line(points, 3)

    r1 = tr.rotate(fi=np.pi / 4) @ final.transpose()
    r2 = tr.shear() @ r1
    r3 = tr.perenos() @ r2
    r4 = tr.homotetia() @ r3

    dis.display_points(
        [final, r1.transpose(), r2.transpose(), r3.transpose(), r4.transpose()],
        ['Begin stage', 'Rotate stage', 'Shear stage', 'Perenos stage', 'Homotetia stage'])


def task_2_b():
    p1 = np.array([[-1, -1, 1], [-1, 1, 1], [1, 1, 1],
                  [1, -1, 1]], dtype=np.float64)
    f1 = tr.bezie_line(p1, 3)

    p2 = np.array([[-2, 0, 1], [-1, 1, 1], [1, 1, 1],
                  [2, 0, 1]], dtype=np.float64)
    f2 = tr.bezie_line(p2, 3)

    p3 = np.array([[-2, 2, 1], [2, -2, 1], [2, 2, 1],
                  [-2, -2, 1]], dtype=np.float64)
    f3 = tr.bezie_line(p3, 3)

    p4 = np.array([[-4, -4, 1], [-2, 0, 1], [0, -4, 1],
                  [2, 0, 1]], dtype=np.float64)
    f4 = tr.bezie_line(p4, 3)

    dis.display_points([f1, f2, f3, f4])


def task_2_c():

    p = np.array([[-3, 0, 1], [-3, 6, 1], [3, 6, 1], [3, 0, 1]])

    l_window = tr.bezie_line(p, line_order=3)

    l_roof = (tr.homotetia(k=2) @ l_window.transpose()).transpose()

    p = np.array([[-9, -5, 1], [-9, -3, 1], [-9, 0, 1], [-6, 0, 1]])
    l_front = tr.bezie_line(p, line_order=3)

    l_back = (tr.perenos(0, 0) @ tr.reflect(axis='y')
              @ l_front.transpose()).transpose()

    p = np.array([[-6, -5, 1], [-3, -5, 1], [-3, -9, 1], [-6, -9, 1]])

    l_wheel_left_1 = tr.bezie_line(p, line_order=3)
    l_wheel_left_2 = (tr.rotate(fi=np.pi, axis=np.array(
        [-6, -7])) @ l_wheel_left_1.transpose()).transpose()

    l_wheel_right_1 = (tr.reflect(
        'y') @ l_wheel_left_1.transpose()).transpose()
    l_wheel_right_2 = (tr.reflect(
        'y') @ l_wheel_left_2.transpose()).transpose()

    p = np.array([[-3, 0, 1], [3, 0, 1]])
    l_border_window = tr.bezie_line(p, line_order=1)

    l_border_wheels = (tr.perenos(Tx=0, Ty=-5) @ tr.homotetia(k=3)
                       @ l_border_window.transpose()).transpose()

    dis.display_points([l_window,
                        l_roof,
                        l_back, l_front,
                        l_wheel_left_1, l_wheel_left_2,
                        l_wheel_right_1, l_wheel_right_2,
                        l_border_window, l_border_wheels])


def main():

    task_1()
    task_2_a()
    task_2_b()
    task_2_c()

    pass


if __name__ == '__main__':

    import os
    file_path = os.path.realpath(__file__)

    shared_path = os.path.join(os.path.dirname(file_path), '..', 'shared')
    import sys
    sys.path.insert(0, shared_path)

    import display as dis
    import transforms as tr
    main()
