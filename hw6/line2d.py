
import numpy as np
from point2d import Point2d, NegInfPoint2d, PosInfPoint2d
EPS = 1e-4


class Line2d:

    def __init__(self, A, B, C) -> None:
        self.A = A
        self.B = B
        self.C = C
        pass

    @staticmethod
    def from_points(a: Point2d, b: Point2d) -> 'Line2d':
        A = b.y() - a.y()
        B = a.x() - b.x()
        C = -A * a.x() - B * a.y()
        self = Line2d(A, B, C)
        return self

    @staticmethod
    def from_point_dir(p: Point2d, dir: np.ndarray) -> 'Line2d':
        return Line2d.from_points(p, Point2d(p.cord + dir))

    def normal_vec(self) -> np.ndarray:
        return np.array([self.A, self.B])

    @staticmethod
    def from_segment(seg: 'Segment2d') -> 'Line2d':
        return Line2d.from_points(seg.a, seg.b)

    def segment_intersection(self, segment: 'Segment2d') -> Point2d:
        rhs = Line2d.from_segment(segment)
        inter_point = self.line_intersection(rhs)

        if type(inter_point) is NegInfPoint2d:
            inter_point = segment.b
        elif type(inter_point) is not PosInfPoint2d and not segment.contains(inter_point):
            inter_point = PosInfPoint2d()
        return inter_point

    def line_intersection(self, rhs: 'Line2d') -> Point2d:
        cross_prod_norms = self.A * rhs.B - self.B * rhs.A

        if np.isclose(cross_prod_norms, 0, atol=EPS):
            if np.isclose(rhs.C, self.C, 0, atol=EPS):
                inter_point = NegInfPoint2d()
            else:
                inter_point = PosInfPoint2d()

        else:
            inter_point = Point2d(np.array([(rhs.C * self.B - self.C * rhs.B) / cross_prod_norms,
                                            (rhs.A * self.C - self.A * rhs.C) / cross_prod_norms]))

        return inter_point

    def sign(self, point: Point2d) -> int:
        val = self.A * point.x() + self.B * point.y() + self.C
        if np.isclose(val, 0, atol=EPS):
            return 0
        elif val < 0:
            return -1
        else:
            return 1

    def contains(self, p: Point2d):
        return self.sign(p) == 0
    def __repr__(self) -> str:
        if self.B != 0:
            return f'y = {-self.A/self.B} * x + {-self.C / self.B}'
        else:
            return f'x={-self.C / self.A}'
        
EPS = 1e-4


class Segment2d:
    def __init__(self, a: Point2d, b: Point2d):
        self.a, self.b = (a, b) if a.x() < b.x() else (b, a)

    def center(self) -> Point2d:
        return Point2d([(self.a.x() + self.b.x()) / 2, (self.a.y() + self.b.y()) / 2])

    def point_in_box(self, point: Point2d):
        lower_x = min(self.a.x(), self.b.x())  # type: ignore
        upper_x = max(self.a.x(), self.b.x())  # type: ignore
        lower_y = min(self.a.y(), self.b.y())  # type: ignore
        upper_y = max(self.a.y(), self.b.y())  # type: ignore

        return lower_x <= point.x() and point.x() <= upper_x and lower_y <= point.y() and point.y() <= upper_y

    def contains(self, p: Point2d) -> bool:

        cur_line = Line2d.from_segment(self)
        return cur_line.contains(p) and self.point_in_box(p)

    def normal_vec(self) -> np.ndarray:
        return Line2d.from_segment(self).normal_vec()

    def get_point_intersection(self, rhs: 'Segment2d') -> Point2d:

        self_line = Line2d.from_segment(self)
        rhs_line = Line2d.from_segment(rhs)
        inter_point = self_line.line_intersection(rhs_line)

        if type(inter_point) is NegInfPoint2d or type(inter_point) is PosInfPoint2d:
            inter_point = rhs.b
            if self.contains(rhs.b):
                inter_point = rhs.b
            elif self.contains(rhs.a):
                inter_point = rhs.a
            elif rhs.contains(self.b):
                inter_point = self.b
            elif rhs.contains(self.a):
                inter_point = self.a
            else:
                inter_point = PosInfPoint2d()
        elif not (self.contains(inter_point) and rhs.contains(inter_point)):
            inter_point = PosInfPoint2d()

        return inter_point

    def looks_at(self, rhs: 'Segment2d') -> bool:
        looks_at = False

        self_vec = self.b.as_array() - self.a.as_array()
        rhs_vec = rhs.b.as_array() - rhs.a.as_array()

        cross_prod = np.cross(self_vec, rhs_vec)
        side = np.cross(rhs_vec,
                        np.array([self.b.x() - rhs.a.x(), self.b.y() - rhs.a.y()]))
        if cross_prod < 0 and side < 0 or cross_prod > 0 and side > 0:
            looks_at = True
        elif np.isclose(cross_prod, 0, atol=EPS):
            looks_at = rhs.contains(self.b)
        return looks_at

    def as_vector(self) -> np.ndarray:
        return np.array([self.b.x() - self.a.x(), self.b.y() - self.a.y()])

   