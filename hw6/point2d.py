import numpy as np

EPS = 10 ** -5
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
    def __eq__(self, rhs) -> bool:
        if type(rhs) is Point2d:
            return bool(np.all(np.isclose(self.cord, rhs.cord, EPS)))
        return False
    def __ne__(self, rhs) -> bool:
        return not self.__eq__(rhs)
    
    

class NegInfPoint2d(Point2d):
    def __init__(self):
        Point2d.__init__(self, None)

    def __repr__(self) -> str:
        return 'NegInfPoint2d'

class PosInfPoint2d(Point2d):
    def __init__(self):
        Point2d.__init__(self, None)

    def __repr__(self) -> str:
        return 'PosInfPoint2d'
    
    
   