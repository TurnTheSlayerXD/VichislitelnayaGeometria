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

            if type(inter_point) is not PosInfPoint2d:
                convex_points.append(inter_point.as_array())
            if halfplane.sign(point) == halfplane.sign(Point2d(self.vertices[i])):
                convex_points.append(self.vertices[i])
        
        # print(f'convex_points = {convex_points}')
        # print(f'point = {point}')
        # print(f'halfplane = {halfplane}')
        # print()


        convex_points = grakham_convex(np.array(convex_points))
        return Polygon(convex_points)
        
        

    def intersect_with_poly(self, rhs: 'Polygon') -> 'Polygon':
        ps = sutherland_hodgman(self.vertices, rhs.vertices)
        arr = [ps, self.vertices, rhs.vertices]
        return Polygon(ps)


def get_half_plane_intersection(cur_point: Point2d, halfplanes: list[Line2d], borderbox: Polygon) -> Polygon:
    if len(halfplanes) <= 1:
        return borderbox.intersect_with_halfplane(cur_point, halfplanes[0])
    middle = len(halfplanes) // 2
    lhs = get_half_plane_intersection(
        cur_point, halfplanes[0:middle], borderbox)
    rhs = get_half_plane_intersection(
        cur_point, halfplanes[middle:], borderbox)
    
    
    inter_poly= lhs.intersect_with_poly(rhs)

    # print(inter_poly.vertices)
    
    # plt.plot(lhs.vertices[:, 0], lhs.vertices[:, 1], color='blue')
    # plt.plot(lhs.vertices[(-1,0), 0], lhs.vertices[(-1,0), 1], color='blue')
    
    # plt.plot(rhs.vertices[:, 0], rhs.vertices[:, 1], color='orange')
    # plt.plot(rhs.vertices[(-1,0), 0], rhs.vertices[(-1,0), 1], color='orange')
    
    # plt.plot(inter_poly.vertices[:, 0], inter_poly.vertices[:, 1], color='red')
    # plt.plot(inter_poly.vertices[(-1,0), 0], inter_poly.vertices[(-1,0), 1], color='red')
    
    
    plt.show()
    
    # assert len(inter_poly.vertices) >= 4
    
    return inter_poly

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
            # VoronoiDiagram.display_locus(cur_locus, ps, bb)
            locuses.append(cur_locus)

        return VoronoiDiagram(locuses)

    def plt_display(self, border_box: np.ndarray):
        plt.grid()
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')

        for locus in self.locuses:
            if len(locus.region.vertices > 0):
                plt.plot(locus.region.vertices[:, 0], locus.region.vertices[:, 1])
                plt.plot(locus.region.vertices[(-1,0), 0], locus.region.vertices[(-1,0), 1])
            plt.scatter([locus.site.x()], [locus.site.y()])

        plt.plot(border_box[:,0], border_box[:,1], color='blue')
        plt.plot(border_box[(-1,0),0], border_box[(-1,0),1], color='blue')
      
        plt.show()


def borderbox_from_points(points: np.ndarray) -> np.ndarray:
    
    x_min = min(points, key=lambda p : p[0])[0] - 1
    x_max = max(points, key=lambda p : p[0])[0] + 1
    
    y_min = min(points, key=lambda p : p[1])[1] - 1
    y_max = max(points, key=lambda p : p[1])[1] + 1
    
    return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])


def gen_points(count, sqr_size):
    res = np.random.random_sample((count, 2)) * sqr_size
    return res


def main():
    np.seterr(all='raise')
    points = gen_points(10, 200)
    # points = np.array([[139.54343688,  59.62712917],
    #                 [109.34011209 , 43.47010423],
    #                 [ 58.82536292, 176.75898259],
    #                 [ 12.69652786, 191.37408475],
    #                 [ 30.06396949, 110.16185165],
    #                 [ 81.00592634,  57.10793624],
    #                 [163.07741421,  28.07121078],
    #                 [186.98378668 , 66.79453036],
    #                 [179.49487108, 135.34627561],
    #                 [ 99.71726674, 105.05841281],])
    
    print(points)
    
    # points = np.array([[0, 1], [1, 3], [2,  2], [3, 0], [6, 8]], dtype=np.float64)
    # points = np.array([ [1, 0], [2, 0]] ,dtype=np.float64)
    borderbox = borderbox_from_points(points)

    diagram = VoronoiDiagram.from_points(points, borderbox)

    diagram.plt_display(borderbox)

    pass


if __name__ == '__main__':
    
    
    
    # print(Point2d([0,1]) != Point2d([0,1])) 
    
    main()





# Failures
 # points = np.array([[ 98.181779,    58.42919726],
    #                     [106.1883763 ,  18.19688011],
    #                     [186.60677075, 102.04045121],
    #                     [ 73.59406036, 161.39945519],
    #                     [194.12245575, 127.79241211],
    #                     [ 75.08986996, 155.65355811],
    #                     [ 62.78794585 ,175.52340751],
    #                     [ 89.57329042, 106.68638621],
    #                     [133.51096216 ,177.50375721],
    #                     [110.07678984 , 95.93537674],])
    
  # points = np.array([[40.53868081 ,16.81670574],
    #                     [50.87877847, 44.77906895],
    #                     [94.41595221, 26.14659082],
    #                     [72.90740551, 54.82827167],
    #                     [ 4.96244047, 97.43623469],
    #                     [54.46548178, 98.95442426],
    #                     [30.10848907 ,54.5129292 ],
    #                     [14.52959804, 99.04380064],
    #                     [15.20834368 ,26.5344677 ],
    #                     [35.12431502, 26.70405088]])