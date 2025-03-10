
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


class Evolute:

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def x(self, t):
        return (self.a ** 2 - self.b ** 2) / self.a * np.cos(t) ** 3

    def y(self, t):
        return (self.b ** 2 - self.a ** 2) / self.b * np.sin(t) ** 3

    def prime(self, t):
        return self.a / self.b * np.tan(t)
    

def get_tangent_of_fig(e, t, prev: np.ndarray, step) -> np.ndarray:
    x = e.x(t)
    y = e.y(t)
    prime = e.prime(t)

    if not prime is None:
        k = prime
        c = -prime * x + y 
        a = seg_intersect(prev[0], prev[1],
                          np.array([x, k * x + c]), np.array([x + step, k * (x + step) + c]))
        
        l = tr.bezie_line(np.array([[a[0], a[1], 1],
                                     [x, k * x + c, 1]]), line_order=1)
    else:
        a = seg_intersect(prev[0], prev[1], np.array([x, y + step]), np.array([x, y - step ]))
        
        l = tr.bezie_line(np.array([[a[0], a[1], 1],
                                     [x, y + step, 1]]), line_order=1)

    return l


def task_1(elipse: Elipse=Elipse(3, 1), n=100):
  
    step = 2 * np.pi / n
    arr = np.array([[0, 0, 0]], dtype=np.float64)
    prev = np.array([ [0, 0], [1, 0]])
    for fi in np.arange(0, np.pi / 2, step):
        line = get_tangent_of_fig(elipse, fi, prev, step)
        arr = np.concatenate((arr, line))
        prev = np.array([[line[0][0], line[0][1]], [line[1][0], line[1][1]]])

    q1 = arr
    q2 = (tr.reflect('y') @ q1.T).T
    q3 = (tr.reflect('x') @ q2.T).T
    q4 = (tr.reflect('y') @ q3.T).T
    l1 = np.concatenate((q1, q2, q3, q4))
    
    evolute = Evolute(elipse.a, elipse.b)
    step = 2 * np.pi / n 
    
    arr = np.array([[0, 0, 0]], dtype=np.float64)
    prev = np.array([ [0, 0], [1, 0]])
    
    for fi in np.arange(0, np.pi / 2, step):
        line = get_tangent_of_fig(evolute, fi, prev, step + 1)
        arr = np.concatenate((arr, line))
        prev = np.array([[line[0][0], line[0][1]], [line[1][0], line[1][1]]])

    q1 = arr
    q2 = (tr.reflect('y') @ q1.T).T
    q3 = (tr.reflect('x') @ q2.T).T
    q4 = (tr.reflect('y') @ q3.T).T
    l2 = np.concatenate((q1, q2, q3, q4))
    
    dis.display_points([l1, l2])


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


def task_2(n=100):
    step = 2 * np.pi / n
    arr = np.array([[0, 0, 0]], dtype=np.float64)

    spiral = ArchimedSpiral(1)
    prev = np.array([ [0, 0], [1, 0]])
    for fi in np.arange(0, 100, step):
        line = get_tangent_of_fig(spiral, fi, prev, step)
        arr = np.concatenate((arr, line))
        prev = np.array([[line[0][0], line[0][1]], [line[1][0], line[1][1]]])
    l1 = arr
    dis.display_points([ l1])
    
    arr = np.array([[0, 0, 0]], dtype=np.float64)
    prev = np.array([ [0, 0], [1, 0]])
    spiral = LogSpiral(0.01, 0.15)
    for fi in np.arange(0, 100, step):
        line = get_tangent_of_fig(spiral, fi, prev, step)
        arr = np.concatenate((arr, line))
        prev = np.array([[line[0][0], line[0][1]], [line[1][0], line[1][1]]])
    
    l2 = arr
    
    dis.display_points([ l2])
    

def main():
    task_1()
    
    task_2()
    pass


if __name__ == '__main__':
    sys.path.insert(0, 'C:/Users/Professional/Desktop/VichislitelnayaGeometria/shared')
    import display as dis 
    import transforms as tr 
    main()
