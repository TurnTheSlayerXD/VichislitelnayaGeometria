
import numpy as np
import sys

EPS = 10 ** -5


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def intersect_param(seg1, seg2):
    A, B = seg1
    C, D = seg2
    dx1 = B[0] - A[0]
    dy1 = B[1] - A[1]
    dx2 = D[0] - C[0]
    dy2 = D[1] - C[1]
    denom = dx1 * dy2 - dy1 * dx2
    if denom == 0:
        return None
    dx = C[0] - A[0]
    dy = C[1] - A[1]
    t = (dx * dy2 - dy * dx2) / denom
    u = (dx * dy1 - dy * dx1) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        return np.array([A[0] + t * dx1, A[1] + t * dy1])
    return None


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


def intersect_lines(k1, c1, k2, c2):
    if np.isclose(k1, k2, atol=10**-3):
        return None, None
    x = (c2 - c1) / (k1 - k2)
    y = k1 * x + c1
    return x, y

def get_tangent_of_fig(fig, t, prev_k, prev_c, step=0.3):
    x = fig.x(t)
    y = fig.y(t)
    prime = fig.prime(t)
    
    if not prime is None:
        k = prime
        c = -prime * x + y
        if prev_k is not None:
            p_x, p_y = intersect_lines(prev_k, prev_c, k, c)
        else:
            p_x, p_y = x - step, k *(x - step) + c
        if p_x is not None:
            l = tr.bezie_line(np.array([[p_x, p_y, 1],
                                        [x, k * x + c, 1]]), line_order=1)
        else:
            l = tr.bezie_line(np.array([[x - step, k * (x - step) + c, 1],
                                        [x, k * x + c, 1]]), line_order=1)
        return l, k, c
    else:
        l = tr.bezie_line(np.array([[x, y - step,1],
                                    [x, y + step, 1]]), line_order=1)

        return l, None, None


def task_1_elipse(n=100.):
    elipse: Elipse = Elipse(10, 4)
    step = 2 * np.pi / n
    arr = []
    k_prev, c_prev = None, None
    for fi in np.arange(0, np.pi / 2, step):
        line, k_prev, c_prev = get_tangent_of_fig(elipse, fi, k_prev, c_prev, step)
        arr.extend(line)

    q1 = np.array(arr)
    q2 = (tr.reflect('y') @ q1.T).T
    q3 = (tr.reflect('x') @ q2.T).T
    q4 = (tr.reflect('y') @ q3.T).T
    l1 = np.concatenate((q1, q2, q3, q4))

    elipse_evolute = ElipseEvolute(elipse.a, elipse.b)
    step = 2 * np.pi / n

    arr = []
    prev = np.array([elipse_evolute.x(0), elipse_evolute.y(0), 1])

    k_prev, c_prev = None, None
    for fi in np.arange(0, np.pi / 2, step):
        line,k_prev, c_prev = get_tangent_of_fig(elipse_evolute, fi, k_prev, c_prev)
        arr.extend(line)

    q1 = np.array(arr)
    q2 = (tr.reflect('y') @ q1.T).T
    q3 = (tr.reflect('x') @ q2.T).T
    q4 = (tr.reflect('y') @ q3.T).T
    l2 = np.concatenate((q1, q2, q3, q4))

    dis.display_points([l1, l2], ['Elipse', 'Elipse Evolute'])


def task_1_hyperbole(n=100):
    hyperbole = Hyperbole(3, 3)
    step = 2 * np.pi / n
    arr = []
    k_prev, c_prev = None, None
    for fi in np.arange(0, np.pi / 2, step):
        line, k_prev, c_prev = get_tangent_of_fig(hyperbole, fi, k_prev, c_prev)
        print(k_prev, c_prev)
        arr.extend(line)

    q1 = np.array(arr)
    q2 = (tr.reflect('y') @ q1.T).T
    q3 = (tr.reflect('x') @ q2.T).T
    q4 = (tr.reflect('y') @ q3.T).T
    l1 = np.concatenate((q1, q2, q3, q4))
    
    print(l1)

    # hyperbole_evolute = HyperboleEvolute(hyperbole.a, hyperbole.b)
    # step = 2 * np.pi / n
    # arr = []
    # k_prev, c_prev = None, None
    # for fi in np.arange(0, np.pi / 4, step):
    #     line, k_prev, c_prev = get_tangent_of_fig(hyperbole_evolute, fi, k_prev, c_prev)
    #     arr.extend(line)

    # q1 = np.array(arr)
    # q2 = (tr.reflect('y') @ q1.T).T
    # q3 = (tr.reflect('x') @ q2.T).T
    # q4 = (tr.reflect('y') @ q3.T).T
    # l2 = np.concatenate((q1, q2, q3, q4))

    dis.display_points([l1], ['Hyperbole', 'Hyperbole Evolute'])


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
        
        l = tr.bezie_line(np.array([[x-step, k * (x - step) + c , 1],
                                    [x+step, k * (x + step) + c, 1]]), line_order=1)
    else:
        l = tr.bezie_line(np.array([[x, y-1, 1],
                                    [x, y+1, 1]]), line_order=1)

    return l

def task_2(n=50):
    step = 2 * np.pi / n
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

    dis.display_points([l1, tang, norm], ['Archimed Spiral', 'Archimed Spiral tangent', 'Archimed Spiral normal'])

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

    dis.display_points([l2, tang, norm], ['Log Spiral', 'Log Spiral tangent', 'Log Spiral normal'])




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

    dis.display_points([l1, tang, norm], ['Rosa', 'Rosa Tangent', 'Rosa Normal'])


def main():
    n = 20
    
    task_1_elipse()
    task_1_hyperbole()

    task_2()
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
