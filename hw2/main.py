
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
        # if np.isclose(np.cos(t), 0, atol=10**-4) or np.isclose(np.tan(t), 0, atol=10**-4):
        #     return None
        if np.isclose(np.tan(t), 0, atol=10**-4):
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


class Line:

    def __init__(self, k=float | None, c=float | None, x=float | None):

        self.k = k
        self.c = c
        self.x = x

    def intersect(self, rhs: 'Line'):
        if self.k is None and rhs.k is None:
            if np.isclose(self.x, rhs.x, atol=10**-4):
                return np.array([self.x, 0])
            raise RuntimeError()
        if self.k is None:
            return np.array([self.x, rhs.k * self.x + rhs.c])
        if rhs.k is None:
            return rhs.intersect(self)
        x = (rhs.c - self.c) / (self.k - rhs.k)
        y = rhs.k * x + rhs.c
        return np.array([x, y])


def get_tang_real(fig, t, step, prev_line: Line | None) -> tuple[np.ndarray, Line]:
    x = fig.x(t)
    y = fig.y(t)
    prime = fig.prime(t)
    k = prime

    if k is None:
        line = Line(None, None, x)
    else:
        c = -k * x + y
        line = Line(k, c)
    
    if prev_line is None:
        if t == 0:
            print('hi', x - step, x + step)
        if k is None:
            return np.array([[x, y - step], [x, y + step]]), line
        else:
            return np.array([[x-step, k*(x-step) + c], [x+step, k*(x+step) + c]]), line
        
    inter = line.intersect(prev_line)
    if t == 3 * np.pi / 2:
        print(k)

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
    elipse: Elipse = Elipse(3, 6)
    step = 10 ** -2
    arr = []
    prev = np.array([elipse.x(0), elipse.y(0), 1])
    for fi in np.arange(0, 2 * np.pi, step):
        arr.append([elipse.x(fi), elipse.y(fi), 1])
    l1 = np.array(arr)

    step = 2 * np.pi / 10
    
    (seg, prev) = get_tang_real(elipse, 0, 100, None)
    segs = [seg]
    step = np.pi / 5
    print(segs)
    start_line = prev
    for fi in np.arange(step, 2 * np.pi, step):
        (seg, prev) = get_tang_real(elipse, fi, 100, prev)
        segs[-1][1] = seg[0]
        segs.append(seg)
    end_line = prev

    p = start_line.intersect(end_line)
    segs[0][0] = p    
    segs[-1][1] = p
    
    elipse_evolute = ElipseEvolute(elipse.a, elipse.b)
    
    arr = []
    step = 10**-1
    for fi in np.arange(-10, 10, step):
        arr.append([elipse_evolute.x(fi), elipse_evolute.y(fi), 1])

    l2 = np.array(arr)

    colors = np.array(['black', 'red', 'orange', 'yellow',
                      'green', 'blue', 'violet'])
    plt.grid()
    for i, batch in enumerate([l1, l2]):
        plt.plot(batch[:, 0], batch[:, 1])
    for i, batch in enumerate(segs):
        plt.plot(batch[:, 0], batch[:, 1], linewidth=3.)
        
    plt.xlim([-10,10])
    plt.ylim([-10,10])
    plt.show()


def task_1_hyperbole(n=25):
    hyperbole = Hyperbole(5., 2.)
    step = 10 ** -2
    arr = []
    for fi in np.arange(-10, 10, step):
        arr.append([hyperbole.x(fi), hyperbole.y(fi), 1])

    l1 = np.array(arr)
    l1_2 = (tr.reflect('y') @ l1.T).T

    step = 1
    
    start = -8
    end = 8
    (seg, prev) = get_tang_real(hyperbole, start, 100, None)
    segs = [seg]
    step = 1
    for fi in np.arange(start + step, end, step):
        (seg, prev) = get_tang_real(hyperbole, fi, 100, prev)
        segs[-1][1] = seg[0]
        segs.append(seg)
        
    for i,seg in enumerate(segs):
        segs[i] = np.hstack((seg, np.ones((seg.shape[0],1))))
    
    segs_1 = [(tr.reflect('y') @ seg.T).T for seg in segs]    
    
    hyperbole_evolute = HyperboleEvolute(hyperbole.a, hyperbole.b)
    step = 10 ** -2
    arr = []
    for fi in np.arange(-1, 1, step):
        arr.append([hyperbole_evolute.x(fi), hyperbole_evolute.y(fi), 1])
    l2 = np.array(arr)
    l2_2 = (tr.reflect('y') @ l2.T).T

    colors = np.array(['black', 'red', 'orange', 'yellow',
                      'green', 'blue', 'violet'])
    plt.grid()
    for i, batch in enumerate([l1,l1_2, l2,l2_2]):
        plt.plot(batch[:, 0], batch[:, 1])
    for i, batch in enumerate(segs + segs_1):
        plt.plot(batch[:, 0], batch[:, 1], linewidth=3.)
        
    plt.xlim([-50,50])
    plt.ylim([-20,20])
    plt.show()



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

    task_2(n)
    task2_rosa(500)
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
