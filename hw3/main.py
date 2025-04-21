

import numpy as np

import matplotlib.pyplot as plt



import matplotlib.pyplot as plt

def task_1(A=np.array([0, 0, 1]), B=np.array([15, 10, 1])):
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

    img = np.zeros((20, 20, 3), dtype=np.uint8)

    for p in points:
        img[p[0],p[1]] = [255, 0, 0]
    
    for p in points2:
        img[p[0],p[1]] = [0, 255, 0]


    # Отображаем результат
    plt.imshow(img)
    plt.axis('off')
    plt.show()



def display_points(points: list, ps, labels=['unknown']):
    colors = np.array(['black', 'red', 'orange', 'yellow', 'green', 'blue', 'violet'])
    plt.grid()

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    for i, line in enumerate(points):
        label=labels[i % len(labels)]
        if label != 'unknown':
            plt.plot(line[:, 0], line[:, 1], 'o', color=colors[i % len(colors)], markersize=1,
                    label=label)
        else:
            plt.plot(line[:, 0], line[:, 1], 'o', color=colors[i % len(colors)], markersize=1)
    
    for i, batch in enumerate(ps):
        plt.scatter(batch[:, 0], batch[:, 1], color=colors[i % len(colors)])
        
    plt.legend(loc='best')

    plt.show()


def task_2_a():
    points = np.array([[-1, -1, 1], [-1, 1, 1], [1, 1, 1]], dtype=np.float64)

    final = tr.bezie_line(points, 2)

    r1 = (tr.rotate(fi=np.pi / 4) @ points.T).T
    r2 = (tr.shear() @ r1.T).T
    r3 = (tr.perenos() @ r2.T).T
    r4 = (tr.homotetia() @ r3.T).T

    display_points(
        [final, tr.bezie_line(r1, 2),  tr.bezie_line(r2, 2),  tr.bezie_line(r3, 2),  tr.bezie_line(r4, 2)],
        [points, r1, r2, r3, r4],
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

    display_points([f1, f2, f3, f4],[p1, p2, p3, p4])


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
