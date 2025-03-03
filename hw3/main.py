

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
        fin_colors += list(np.repeat(color, n_vertices))

    print(fin_colors)
    for i, poly in enumerate(polygons):
        plt.plot(poly[:, 0], poly[:, 1], 'o', color=fin_colors[i], markersize=1)

    plt.show()


def perenos(Tx=2, Ty=0):
    mat = np.array([[1, 0, Tx], [0, 1, Ty], [0, 0, 1]])
    return mat


def rotate(fi=np.pi / 2, axis=np.array([0, 0])):
    mat_forw = perenos(-axis[0], -axis[1])
    
    mat_rot = np.array([[np.cos(fi), np.sin(fi), 0], [-np.sin(fi), np.cos(fi), 0], [0, 0, 1]])

    mat_back = perenos(axis[0], axis[1]) 

    res = mat_back @ mat_rot @ mat_forw

    return res


def homotetia(axis=np.array([0, 0]), k=2):
    mat = k * np.array([ [1, 0, -axis[0]], [0, 1, -axis[1]], [0, 0, 1 / k] ])
    res = perenos(axis[0], axis[1]) @ mat   
    return res


def shear(alpha=np.pi / 3, beta=np.pi / 3):
    mat = np.array([[1, np.tan(alpha), 0], [np.tan(beta), 1, 0], [0 , 0 , 1]])
    return mat


def task_1(A=np.array([-10, -7, 1]), B=np.array([15, 23, 1])):

    dx = B[0] - A[0]
    dy = B[1] - A[1]

    n_points = dx + 1
    points = np.array([ [i, 0, 1] for i in range(A[0], B[0] + 1) ])
    
    points[0] = A
    di_prev = 2 * (points[0][0] * dy - points[0][1] * dx) + 2 * dy - dx
    points[1][1] = points[0][1] if di_prev < 0 else points[0][1] + 1  

    for i in range(2, n_points):
        di = di_prev + 2 * dy - (points[i - 1][1] - points[i - 2][1]) * dx
        points[i][1] = points[i - 1][1] if di_prev < 0 else points[i - 1][1] + 1  
        di_prev = di

    display(np.array([points]))


def task_2(line_order: int=2, points=np.array([[-1, -1, 1], [0, 1, 1], [1, 1, 1]], dtype=np.float64)):

    final = []
    for t in np.arange(0, 1.1, 0.01):
        cur_points = points
        for i in range(line_order):
            new_points = np.array([[0, 0, 1] for _ in range(len(cur_points) - 1)], dtype=np.float64)
            for i in range(1, len(cur_points)):
                new_points[i - 1] = (1 - t) * cur_points[i - 1] + t * cur_points[i] 
            cur_points = new_points

        final.append(np.array(cur_points[0]))

    final = np.array(final)

    r1 = rotate(fi=np.pi / 4) @ final.transpose()
    r2 = shear() @ r1
    r3 = perenos() @ r2
    r4 = homotetia() @ r3

    display(np.array([final, r1.transpose(), r2.transpose(), r3.transpose(), r4.transpose()]))


def main():
    task_1()

    task_2()

    pass


if __name__ == '__main__':
    main()
