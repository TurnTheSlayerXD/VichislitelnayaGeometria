import numpy as np
from line2d import Line2d, Segment2d
from point2d import Point2d, NegInfPoint2d, PosInfPoint2d
from grakham import grakham_convex
from sutherland_hodgman import sutherland_hodgman
import matplotlib.patches as dis

import matplotlib.pyplot as plt
colors = np.array(['black', 'red', 'yellow', 'yellow',
                   'green', 'blue', 'violet'])


class Polygon:

    def __init__(self, vertices: np.ndarray = np.array([])) -> None:
        self.vertices = vertices
        self.cur_vertex_ind = 0
        pass

    def intersect_with_halfplane(self, point: Point2d, halfplane: Line2d):
        convex_points = []
        for i in range(len(self.vertices)):
            j = (i + 1) % len(self.vertices)
            cur_side = Segment2d(
                Point2d(self.vertices[i]), Point2d(self.vertices[j]))
            inter_point = halfplane.segment_intersection(cur_side)

            if type(inter_point) is not PosInfPoint2d and type(inter_point) is not NegInfPoint2d:
                convex_points.append(inter_point.as_array())

            if halfplane.sign(point) == halfplane.sign(Point2d(self.vertices[i])):
                convex_points.append(self.vertices[i])
       
        return Polygon(grakham_convex(np.array(convex_points)))

    def intersect_with_poly(self, rhs: 'Polygon') -> 'Polygon':
        ps = sutherland_hodgman(self.vertices, rhs.vertices)

        arr = [ps, self.vertices, rhs.vertices]

        # for i,poly in enumerate(arr):
        #     t = dis.Polygon(poly[:], alpha=0.5,
        #                     color=colors[i % len(colors)])
        #     plt.gca().add_patch(t)
        #     plt.scatter(poly[:,0], poly[:,1])
        #     plt.title('Polygon intersection')

        # plt.show()

        return Polygon(ps)


def get_half_plane_intersection(cur_point: Point2d, halfplanes: list[Line2d], borderbox: Polygon) -> Polygon:
    if len(halfplanes) == 1:
        return borderbox.intersect_with_halfplane(cur_point, halfplanes[0])
    middle = len(halfplanes) // 2
    lhs = get_half_plane_intersection(
        cur_point, halfplanes[:middle], borderbox)
    rhs = get_half_plane_intersection(
        cur_point, halfplanes[middle:], borderbox)
    return lhs.intersect_with_poly(rhs)


class VoronoiLocus:
    def __init__(self, convex: Polygon, point: Point2d):
        self.region = convex
        self.site = point


class VoronoiDiagram:

    def __init__(self, locuses: list[VoronoiLocus]) -> None:
        self.locuses = locuses

    @staticmethod
    def make_locus(site: Point2d, points: list[Point2d], borderbox: Polygon) -> VoronoiLocus:
        halfplanes: list[Line2d] = []

        for cur_point in points:
            if cur_point != site:
                segment = Segment2d(site, cur_point)

                cur_halfplane = Line2d.from_point_dir(
                    segment.center(), segment.normal_vec())
                halfplanes.append(cur_halfplane)

        region = get_half_plane_intersection(site, halfplanes, borderbox)
        locus = VoronoiLocus(region, site)
        return locus

    @staticmethod
    def display_locus(lokus: VoronoiLocus, points: np.ndarray, border_box: np.ndarray):
        poly = lokus.region.vertices
        t = dis.Polygon(poly[:], alpha=0.5,
                        color='red')
        plt.gca().add_patch(t)
        plt.scatter(poly[:, 0], poly[:, 1])
        plt.scatter(points[:, 0], points[:, 1])

        plt.scatter(lokus.site.x(), lokus.site.y())
        
        plt.plot(border_box[:,0], border_box[:,1], color='blue')
        plt.plot(border_box[(-1,0),0], border_box[(-1,0),1], color='blue')
        
        plt.title('Lokus')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.grid()
        plt.show()

    @staticmethod
    def from_points(ps: np.ndarray, bb: np.ndarray):
        points = [Point2d(p) for p in ps]
        borderbox = Polygon(bb)
        locuses: list[VoronoiLocus] = []
        for p in points:
            cur_locus = VoronoiDiagram.make_locus(p, points, borderbox)
            VoronoiDiagram.display_locus(cur_locus, ps, bb)

            locuses.append(cur_locus)
        return VoronoiDiagram(locuses)

    def plt_display(self, border_box: np.ndarray):
        plt.grid()
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')

        for locus in self.locuses:
            plt.plot(locus.region.vertices[:, 0], locus.region.vertices[:, 1])
            plt.scatter([locus.site.x()], [locus.site.y()])

        plt.plot(border_box[:,0], border_box[:,1], color='blue')
        plt.plot(border_box[(-1,0),0], border_box[(-1,0),1], color='blue')
      
        plt.show()


def borderbox_from_points(points: np.ndarray) -> np.ndarray:
    
    x_min = min(points, key=lambda p : p[0])[0] - 1
    x_max = max(points, key=lambda p : p[0])[0] + 1
    
    y_min = min(points, key=lambda p : p[1])[1] - 1
    y_max = max(points, key=lambda p : p[1])[1] + 1
    
    print(f'y_max = {y_max}')
    return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])


def gen_points(count, sqr_size):
    res = np.random.random_sample((count, 2)) * sqr_size
    return res


def main():

    points = gen_points(3, 20)
    # points = np.array([[0, 1], [1, 3], [2,  2], [3, 0], [6, 8]], dtype=np.float64)
    # points = np.array([ [1, 0], [2,  0], [3, 1], [3, 2]])
    borderbox = borderbox_from_points(points)

    diargram = VoronoiDiagram.from_points(points, borderbox)

    diargram.plt_display(borderbox)

    pass


if __name__ == '__main__':
    main()
