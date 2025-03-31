
import numpy as np
import sys

EPS = 10 ** -5


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def seg_intersect(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)

    if denom == 0:
        denom = EPS

    db[db == 0] = EPS

    return (num / denom + EPS) * db + b1


class Elipse:

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def x(self, t):
        return self.a * np.cos(t)

    def y(self, t):
        return self.b * np.sin(t)

    def prime(self, t):
        if t == np.pi / 2:
            return None
        if np.tan(t) == 0:
            return None
        return -self.b / self.a * 1 / np.tan(t)


class Hyperbole:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def x(self, t):
        return self.a * np.cosh(t)

    def y(self, t):
        return self.b * np.sinh(t)

    def prime(self, t):
        return self.b * np.cosh(t) / (self.a * np.sinh(t) + EPS)


class ElipseEvolute:

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def x(self, t):
        return (self.a ** 2 - self.b ** 2) / self.a * np.cos(t) ** 3

    def y(self, t):
        return (self.b ** 2 - self.a ** 2) / self.b * np.sin(t) ** 3

    def prime(self, t):
        return self.a / self.b * np.tan(t)


class HyperboleEvolute:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def x(self, t):
        return self.a * np.cosh(t) ** 3 * (1 + (self.b / self.a) ** 2)

    def y(self, t):
        return self.b * np.sinh(t) ** 3 * (1 + (self.a / self.b) ** 2)

    def prime(self, t):
        return self.b * (1 + (self.a / self.b) ** 2) \
            / (self.a * (1 + (self.b / self.a) ** 2)) * np.sinh(t) / np.cosh(t)


def get_tangent_of_fig(fig, t, prev: np.ndarray, step) -> np.ndarray:
    x = fig.x(t)
    y = fig.y(t)
    prime = fig.prime(t)

    if not prime is None:
        k = prime
        c = -prime * x + y
        l = tr.bezie_line(np.array([prev,
                                    [x, k * x + c, 1]]), line_order=1)
    else:
        l = tr.bezie_line(np.array([prev,
                                    [x, y + step, 1]]), line_order=1)

    return l


def task_1_elipse():
    elipse: Elipse = Elipse(10, 4)
    n = 10
    step = 2 * np.pi / n
    arr = []
    prev = np.array([elipse.x(0), elipse.y(0), 1])
    for fi in np.arange(0, np.pi / 2, step):
        line = get_tangent_of_fig(elipse, fi, prev, step)
        arr.extend(line)
        prev = line[-1]

    q1 = np.array(arr)
    q2 = (tr.reflect('y') @ q1.T).T
    q3 = (tr.reflect('x') @ q2.T).T
    q4 = (tr.reflect('y') @ q3.T).T
    l1 = np.concatenate((q1, q2, q3, q4))

    elipse_evolute = ElipseEvolute(elipse.a, elipse.b)
    step = 2 * np.pi / n

    arr = []
    prev = np.array([elipse_evolute.x(0), elipse_evolute.y(0), 1])

    for fi in np.arange(0, np.pi / 2, step):
        line = get_tangent_of_fig(elipse_evolute, fi, prev, step + 1)
        arr.extend(line)
        prev = line[-1]

    q1 = np.array(arr)
    q2 = (tr.reflect('y') @ q1.T).T
    q3 = (tr.reflect('x') @ q2.T).T
    q4 = (tr.reflect('y') @ q3.T).T
    l2 = np.concatenate((q1, q2, q3, q4))

    dis.display_points([l1, l2], ['Elipse', 'Elipse Evolute'])


def task_1_hyperbole():
    n = 10

    hyperbole = Hyperbole(10, 10)
    step = 2 * np.pi / n
    arr = []
    prev = np.array([hyperbole.x(0), hyperbole.y(0), 1])
    for fi in np.arange(0, np.pi / 2, step):
        line = get_tangent_of_fig(hyperbole, fi, prev, step)
        arr.extend(line)
        prev = line[-1]

    q1 = np.array(arr)
    q2 = (tr.reflect('y') @ q1.T).T
    q3 = (tr.reflect('x') @ q2.T).T
    q4 = (tr.reflect('y') @ q3.T).T
    l1 = np.concatenate((q1, q2, q3, q4))

    hyperbole_evolute = HyperboleEvolute(hyperbole.a, hyperbole.b)
    step = 2 * np.pi / n
    arr = []
    prev = np.array([hyperbole_evolute.x(0), hyperbole_evolute.y(0), 1])

    for fi in np.arange(0, np.pi / 4, step):
        line = get_tangent_of_fig(hyperbole_evolute, fi, prev, step)
        arr.extend(line)
        prev = line[-1]

    q1 = np.array(arr)
    q2 = (tr.reflect('y') @ q1.T).T
    q3 = (tr.reflect('x') @ q2.T).T
    q4 = (tr.reflect('y') @ q3.T).T
    l2 = np.concatenate((q1, q2, q3, q4))

    dis.display_points([l1, l2], ['Hyperbole', 'Hyperbole Evolute'])


def IsZero(a):
    if np.abs(a) < EPS:
        return True
    return False


class ArchimedSpiral:

    def __init__(self, a):
        self.a = a / np.pi / 2

    def x(self, t):
        return self.a * t * np.cos(t)

    def y(self, t):
        return self.a * t * np.sin(t)

    def prime(self, t):
        EPS = 10 ** -5
        sin = np.sin(t)
        cos = np.cos(t)
        if IsZero(cos - t * sin):
            return None
        return (sin + t * cos) / (cos - t * sin)


class LogSpiral:

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def x(self, t):
        return self.a * np.exp(self.b * t) * np.cos(t)

    def y(self, t):
        return self.a * np.exp(self.b * t) * np.sin(t)

    def prime(self, t):
        EPS = 10 ** -5
        tg = np.tan(t)

        return (self.b * tg + 1) / (self.b - tg)


class Rosa:
    def __init__(self, a, k):
        self.a = a
        self.k = k

    def x(self, t):
        return self.a * np.sin(self.k * t) * np.cos(t)

    def y(self, t):
        return self.a * np.sin(self.k * t) * np.sin(t)

    def prime(self, t):
        k = self.k
        sinkt = np.sin(k * t)
        coskt = np.cos(k * t)
        sint = np.sin(t)
        cost = np.cos(t)

        if IsZero(k * coskt * cost - sinkt * sint):
            return None
        return (k * coskt * sint + sinkt * cost) \
            / (k * coskt * cost - sinkt * sint)


def task_2(n=10):
    step = 2 * np.pi / n
    arr = []
    spiral = ArchimedSpiral(1)
    prev = np.array([spiral.x(0), spiral.y(0), 1])
    for fi in np.arange(0, np.pi * 10, step):
        line = get_tangent_of_fig(spiral, fi, prev, step)
        arr.extend(line)
        prev = line[-1]
    l1 = np.array(arr)
    dis.display_points([l1], ['Archimed Spiral'])

    arr = []
    spiral = LogSpiral(0.01, 0.15)
    prev = np.array([spiral.x(0), spiral.y(0), 1])
    for fi in np.arange(0, np.pi * 10, step):
        line = get_tangent_of_fig(spiral, fi, prev, step)
        arr.extend(line)
        prev = line[-1]

    l2 = np.array(arr)

    dis.display_points([l2], ['Log Spiral'])


def task2_rosa():
    n = 10
    step = 2 * np.pi / n
    arr = []
    rosa = Rosa(6, 5)
    prev = np.array([rosa.x(0), rosa.y(0), 1])

    for fi in np.arange(0, np.pi * 2, step):
        line = get_tangent_of_fig(rosa, fi, prev, step)
        arr.extend(line)
        prev = line[-1]
    l1 = np.array(arr)
    dis.display_points([l1], ['Rosa'])


def main():
    task_1_elipse()
    task_1_hyperbole()

    task_2()
    task2_rosa()
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
