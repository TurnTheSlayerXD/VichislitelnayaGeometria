
import matplotlib.pyplot as plt
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
        if np.isclose(np.tan(t), 0, atol=10**-2):
            return None
        if (np.isclose(np.cos(t), 0, atol=10**-2)):
            return None
        return -self.b / self.a * 1 / np.tan(t)


class Hyperbole:
    def __init__(self, a, b):
        self.a = np.float64(a)
        self.b = np.float64(b)

    def x(self, t):
        return self.a * np.cosh(t)

    def y(self, t):
        return self.b * np.sinh(t)

    def prime(self, t):

        if np.isclose(np.sinh(t), 0., atol=10**-3) or np.isclose(t, 0., atol=10**-3):
            return None
        return self.b * np.cosh(t) / (self.a * np.sinh(t))


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


def intersect_lines(k1, c1, k2, c2, prev_x, step):
    if k1 is None:
        return [prev_x, k2 * prev_x + c2]

    return x, y


class Line:

    def __init__(self, k=float | None, c=float | None, x=float | None):

        self.k = k
        self.c = c
        self.x = x

    def intersect(self, rhs: 'Line'):
        if self.k is None and rhs.k is None:
            raise RuntimeError()
        if self.k is None:
            return np.array([self.x, rhs.k * self.x + rhs.c])
        if rhs.k is None:
            return rhs.intersect(self)
        x = (rhs.c - self.c) / (self.k - rhs.k)
        y = rhs.k * x + rhs.c
        return np.array([x, y])


def get_tang_real(fig, t, step, prev_line: Line) -> tuple[np.ndarray, Line]:
    x = fig.x(t)
    y = fig.y(t)
    prime = fig.prime(t)
    k = prime

    if k is None:
        line = Line(None, None, x)
    else:
        c = -k * x + y
        line = Line(k, c)

    inter = line.intersect(prev_line)

    if k is None:
        return np.array([inter, [x, y + step]]), line
    else:
        return np.array([inter, [x+step, k*(x+step) + c]]), line


def display_points(points: list, rest=[], labels=['unknown']):
    colors = np.array(['black', 'red', 'orange', 'yellow',
                      'green', 'blue', 'violet'])
    plt.grid()
    for i, batch in enumerate(points):
        plt.plot(batch[:, 0], batch[:, 1], label=labels[i % len(labels)])
    for i, batch in enumerate(rest):
        plt.plot(batch[:, 0], batch[:, 1], label=labels[i %
                 len(labels)], linewidth=3.)

    plt.show()


def task_1_elipse(n=50):
    elipse: Elipse = Elipse(3, 5)
    step = 10 ** -2
    arr = []
    prev = np.array([elipse.x(0), elipse.y(0), 1])
    for fi in np.arange(0, 2 * np.pi, step):
        arr.append([elipse.x(fi), elipse.y(fi), 1])
    l1 = np.array(arr)

    step = 2 * np.pi / 10
    segs = []
    prev = Line(1, 0, 0.)
    for fi in np.arange(0, 2 * np.pi, step):
        step_fi = np.cos(fi) * step
        (seg, prev) = get_tang_real(elipse, fi, step_fi, prev)
        print(type(prev))
        segs.append(seg)

    elipse_evolute = ElipseEvolute(elipse.a, elipse.b)

    arr = []
    step = 10**-1
    for fi in np.arange(-10, 10, step):
        arr.append([elipse_evolute.x(fi), elipse_evolute.y(fi), 1])

    l2 = np.array(arr)

    display_points([l1, l2], segs, ['Elipse', 'Elipse Evolute'])


def task_1_hyperbole(n=25):
    hyperbole = Hyperbole(5., 3.)
    step = 10 ** -2
    arr = []
    for fi in np.arange(-2, 2, step):
        arr.append([hyperbole.x(fi), hyperbole.y(fi), 1])

    l1 = np.array(arr)
    l1_2 = (tr.reflect('y') @ l1.T).T

    step = 10 ** -1 * 10
    segs = []
    for fi in np.arange(-2, 2, step):
        step_fi = np.cos(fi) * step
        print(step_fi)
        segs.append(get_tang_real(hyperbole, fi, step_fi, prev))

    hyperbole_evolute = HyperboleEvolute(hyperbole.a, hyperbole.b)
    step = 10 ** -2
    arr = []
    for fi in np.arange(-1, 1, step):
        arr.append([hyperbole_evolute.x(fi), hyperbole_evolute.y(fi), 1])
    l2 = np.array(arr)
    l2_2 = (tr.reflect('y') @ l2.T).T

    display_points([l1, l1_2, l2, l2_2], segs, [
                   'Hyperbole', 'Hyperbole Evolute'])


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
        return self.a * np.cos(self.k * t) * np.cos(t)

    def y(self, t):
        return self.a * np.cos(self.k * t) * np.sin(t)

    def prime(self, t):
        k = self.k
        sinkt = np.sin(k * t)
        coskt = np.cos(k * t)
        sint = np.sin(t)
        cost = np.cos(t)

        if IsZero(-sint * coskt - k * sinkt * cost):
            return None
        return (cost * coskt - k * sinkt * sint) \
            / (-sint * coskt - k * sinkt * cost)


def get_tangent(fig, t):
    x = fig.x(t)
    y = fig.y(t)
    prime = fig.prime(t)

    if not prime is None:
        k = prime
        c = -prime * x + y

        step = 1
        while k * (x + step) + c > y + 5:
            step /= 2
        l = tr.bezie_line(np.array([[x-step, k*(x - step) + c, 1],
                                    [x+step, k*(x + step) + c, 1]]), line_order=1)
    else:
        l = tr.bezie_line(np.array([[x, y-1, 10],
                                    [x, y+1, 1]]), line_order=1)

    return l


def get_normal(fig, t):
    x = fig.x(t)
    y = fig.y(t)
    prime = fig.prime(t)

    if not prime is None and not np.isclose(prime, 0., atol=10**-3):
        k = -1 / prime
        c = -k * x + y

        step = 1
        count = 0
        while k * (x + step) + c > y + 3 and count < 50:
            step /= 2
            count += 1

        l = tr.bezie_line(np.array([[x-step, k * (x - step) + c, 1],
                                    [x+step, k * (x + step) + c, 1]]), line_order=1)
    else:
        l = tr.bezie_line(np.array([[x, y-1, 1],
                                    [x, y+1, 1]]), line_order=1)

    return l


def task_2(n=50):
    step = 0.1
    arr = []
    spiral = ArchimedSpiral(1)
    prev = np.array([spiral.x(0), spiral.y(0), 1])
    for fi in np.arange(0, np.pi * 10, step):
        line = get_tangent_of_fig(spiral, fi, prev, step)
        arr.extend(line)
        prev = line[-1]
    l1 = np.array(arr)
    tang = get_tangent(spiral, np.pi / 2)
    norm = get_normal(spiral, np.pi / 2)

    dis.display_points([l1, tang, norm], ['Archimed Spiral',
                       'Archimed Spiral tangent', 'Archimed Spiral normal'])

    arr = []
    spiral = LogSpiral(0.01, 0.15)
    prev = np.array([spiral.x(0), spiral.y(0), 1])
    for fi in np.arange(0, np.pi * 10, step):
        line = get_tangent_of_fig(spiral, fi, prev, step)
        arr.extend(line)
        prev = line[-1]

    l2 = np.array(arr)
    tang = get_tangent(spiral, np.pi * 8.5)
    norm = get_normal(spiral, np.pi * 8.5)

    dis.display_points([l2, tang, norm], ['Log Spiral',
                       'Log Spiral tangent', 'Log Spiral normal'])


def task2_rosa(n=25):
    step = 2 * np.pi / n
    arr = []
    rosa = Rosa(10, 4)
    prev = np.array([rosa.x(0), rosa.y(0), 1])

    for fi in np.arange(0, np.pi * 2, step):
        line = get_tangent_of_fig(rosa, fi, prev, step)
        arr.extend(line)
        prev = line[-1]

    tang = get_tangent(rosa, 2 * np.pi / 6)
    norm = get_normal(rosa, 2 * np.pi / 6)

    l1 = np.array(arr)
    # dis.display_points([l1], ['Rosa', 'Rosa Tangent', 'Rosa Normal'])

    dis.display_points([l1, tang, norm], [
                       'Rosa', 'Rosa Tangent', 'Rosa Normal'])


def main():
    n = 20

    task_1_elipse(n)
    task_1_hyperbole(n)

    # task_2(n)
    # task2_rosa(500)
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
