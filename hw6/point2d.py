import numpy as np


class Point2d:

    def __init__(self, cord: np.ndarray | list) -> None:
        self.cord = cord
        pass

    def x(self) -> np.float64:
        return self.cord[0]

    def y(self) -> np.float64:
        return self.cord[1]

    def as_array(self) -> np.ndarray:
        return np.array(self.cord)

    def __repr__(self) -> str:

        return f'x={self.cord[0]} y={self.cord[1]}'


class NegInfPoint2d(Point2d):
    def __init__(self):
        Point2d.__init__(self, None)


class PosInfPoint2d(Point2d):
    def __init__(self):
        Point2d.__init__(self, None)
