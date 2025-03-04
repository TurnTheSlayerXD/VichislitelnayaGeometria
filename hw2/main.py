
import numpy as np
import sys


class Elipse:

    def __init__(self, a, b):
        self.a = a
        self.b = b


def get_tangent_of_elipse(a, b, p: np.ndarray, step) -> np.ndarray:
    t = np.arccos(p[0] / a)
    if t != 0:
        prime = -b / a * 1 / np.tan(t)
        k = prime
        c = -prime * p[0] + p[1] 
        l = tr.bezie_line(np.array([[p[0] - step, k * (p[0] - step) + c, 1],
                                     [p[0] + step, k * (p[0] + step) + c, 1]]), line_order=1)
    else:
        l = tr.bezie_line(np.array([[p[0], p[1] - step, 1],
                                     [p[0], p[1] + step, 1]]), line_order=1)

    return l


def task_1(elipse: Elipse=Elipse(10, 2), n=50):

    a = elipse.a
    b = elipse.b
  
    step = np.pi / 2 / n
    arr = np.array([[0, 0, 0]], dtype=np.float64)
    for fi in np.arange(0, np.pi / 2, step):
        p = [a * np.cos(fi), b * np.sin(fi), 1]
        line = get_tangent_of_elipse(a, b, p, np.cos(step) / n)
        arr = np.concatenate((arr, line))

    q1 = arr
    q2 = (tr.reflect('y') @ q1.transpose()).transpose()
    q3 = (tr.reflect('x') @ q2.transpose()).transpose()
    q4 = (tr.reflect('y') @ q3.transpose()).transpose()
    l = np.concatenate((q1, q2, q3, q4))
    # l = q1
    
    dis.display_points([l])


def main():
    
    task_1()
    pass


if __name__ == '__main__':
    sys.path.insert(0, 'C:/Users/Professional/Desktop/VichislitelnayaGeometria/shared')
    import display as dis 
    import transforms as tr 
    main()
