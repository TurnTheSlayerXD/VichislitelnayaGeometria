
import numpy as np


def perenos(Tx=2, Ty=0) -> np.ndarray:
    mat = np.array([[1, 0, Tx], [0, 1, Ty], [0, 0, 1]])
    return mat


def rotate(fi=np.pi / 2, axis=np.array([0, 0])) -> np.ndarray:
    mat_forw = perenos(-axis[0], -axis[1])
    
    mat_rot = np.array([[np.cos(fi), np.sin(fi), 0], [-np.sin(fi), np.cos(fi), 0], [0, 0, 1]])

    mat_back = perenos(axis[0], axis[1]) 

    res = mat_back @ mat_rot @ mat_forw

    return res


def homotetia(axis=np.array([0, 0]), k=2) -> np.ndarray:
    mat = k * np.array([ [1, 0, -axis[0]], [0, 1, -axis[1]], [0, 0, 1 / k] ])
    res = perenos(axis[0], axis[1]) @ mat   
    return res


def shear(alpha=np.pi / 3, beta=np.pi / 3) -> np.ndarray:
    
    mat = np.array([[1, np.tan(alpha), 0], [np.tan(beta), 1, 0], [0, 0, 1]])
    return mat


def bezie_line(points: np.ndarray, line_order: int=2) -> np.ndarray:

    final = []
    d = 10 ** -2
    for t in np.arange(0, 1 + d, d):
        cur_points = points
        for i in range(line_order):
            new_points = np.array([[0, 0, 1] for _ in range(len(cur_points) - 1)], dtype=np.float64)
            for i in range(1, len(cur_points)):
                new_points[i - 1] = (1 - t) * cur_points[i - 1] + t * cur_points[i] 
            cur_points = new_points
        final.append(np.array(cur_points[0]))

    final = np.array(final)

    return final


def reflect(axis='x') -> np.ndarray:
    mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    if axis == 'x':
        mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    elif axis == 'y':
        mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return mat


def elipse(a: 1, b: 2) -> np.ndarray:
    
    step = 0.01
    arr = np.array([[a * np.cos(t), b * np.sin(t), 1]
           for t in np.arange(0, np.pi / 2 + step, step)])
    q1 = arr
    
    q2 = (reflect('y') @ q1.transpose()).transpose()
    q3 = (reflect('x') @ q2.transpose()).transpose()
    q4 = (reflect('y') @ q3.transpose()).transpose()
    l = np.concatenate((q1, q2, q3, q4))
    return l
