import numpy as np
from line2d import Line2d, Segment2d
from point2d import Point2d, NegInfPoint2d, PosInfPoint2d
from grakham import grakham_convex
from sutherland_hodgman import sutherland_hodgman
import matplotlib.patches as dis

import matplotlib.pyplot as plt
colors = np.array(['black', 'red', 'yellow', 'yellow',
                   'green', 'blue', 'violet'])


EPS = 10 ** -5


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
        
        # plt.plot(self.vertices[:,0], self.vertices[:,1], color='blue')
        # plt.plot(self.vertices[(-1,0),0], self.vertices[(-1,0),1], color='blue')
        
        # plt.plot(convex_points[:, 0], convex_points[:, 1], color='red')
        # plt.plot(convex_points[(-1,0), 0], convex_points[(-1,0), 1], color='red')
        # plt.scatter(convex_points[:, 0], convex_points[:, 1], color='red')
        
        # plt.scatter(point.x(), point.y(), color='green')
        
        # plt.title(f'{halfplane}')
        # plt.show()        
        
        
        return Polygon(convex_points)
        
        

    def intersect_with_poly(self, rhs: 'Polygon') -> 'Polygon':
        ps = sutherland_hodgman(self.vertices, rhs.vertices)
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
    
    
    # print('lhs', lhs.vertices)
    # print('rhs', rhs.vertices)
    # print('inter_poly', inter_poly.vertices)

    # a = dis.Polygon(lhs.vertices[:], alpha=0.5,
    #                     color='green')
    # plt.gca().add_patch(a)

    # b = dis.Polygon(rhs.vertices[:], alpha=0.5,
    #                     color='blue')
    # plt.gca().add_patch(b)

    # c = dis.Polygon(inter_poly.vertices[:], alpha=0.5,
    #                     color='red')
    # plt.gca().add_patch(c)
    
    # ax = plt.gca()
    # ax.set_aspect('equal', adjustable='box')
    # plt.grid()
    
    # bb = borderbox.vertices
    # plt.plot(bb[:,0], bb[:,1], color='blue')
    # plt.plot(bb[(-1,0),0], bb[(-1,0),1], color='blue')
   
    # plt.show()
     

    # plt.plot(lhs.vertices[:, 0], lhs.vertices[:, 1], color='blue')
    # plt.plot(lhs.vertices[(-1,0), 0], lhs.vertices[(-1,0), 1], color='blue')
    
    # plt.plot(rhs.vertices[:, 0], rhs.vertices[:, 1], color='orange')
    # plt.plot(rhs.vertices[(-1,0), 0], rhs.vertices[(-1,0), 1], color='orange')
    
    # plt.plot(inter_poly.vertices[:, 0], inter_poly.vertices[:, 1], color='red')
    # plt.plot(inter_poly.vertices[(-1,0), 0], inter_poly.vertices[(-1,0), 1], color='red')
    
    return inter_poly

class VoronoiLocus:
    def __init__(self, convex: Polygon, point: Point2d):
        self.region = convex
        self.site = point

        self.neighbours : list[tuple[VoronoiLocus, Segment2d]] = []
        
        
    def __eq__(self, rhs):
        if type(rhs) is VoronoiLocus:
            return self.site == rhs.site
        return False

    def __ne__(self, rhs):
        return not self.__eq__(rhs) 


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


def find_first_norm(segment : Segment2d, diagram: VoronoiDiagram):
    
    for locus in diagram.locuses:
        vertices = locus.region.vertices
        for i in range(len(vertices)):
            j  = (i + 1) % len(vertices)
            rhs = Segment2d(Point2d(vertices[i]),Point2d(vertices[j]))
            if segment.get_point_intersection(rhs) is not PosInfPoint2d\
                and segment.get_point_intersection(rhs) is not NegInfPoint2d\
                and np.isclose(np.cross(segment.normal_vec(), rhs.as_vector()), 0, atol=10**-10):
                return rhs
    
    return None
    
    
def get_common_side(locus_a : VoronoiLocus, locus_b: VoronoiLocus) -> Segment2d | None:
    edges_a = [[locus_a.region.vertices[i], 
                   locus_a.region.vertices[ (i + 1) % len(locus_a.region.vertices)] ] 
               for i in range(len(locus_a.region.vertices))]
    edges_b = [[locus_b.region.vertices[i], 
                   locus_b.region.vertices[ (i + 1) % len(locus_b.region.vertices)] ] 
               for i in range(len(locus_b.region.vertices))]
    for a in edges_a:
        for b in edges_b:
            if np.all(np.isclose(a[0], b[0], atol=EPS)) and np.all(np.isclose(a[1] , b[1], atol=EPS)) or\
                np.all(np.isclose(a[0], b[1], atol=EPS)) and np.all(np.isclose(a[1] , b[0], atol=EPS)):
                    return Segment2d(Point2d(a[0]), Point2d(b[0]))
    return None
    

def delone_triangulation(diagram : VoronoiDiagram) -> list:
    triangulation = []
    locuses = diagram.locuses
        
    for i in range(len(locuses) - 1):
        for j in range(i + 1, len(locuses)):
            
            common_side = get_common_side(locuses[i], locuses[j])
            if common_side is not None:
                
                if (locuses[j], common_side) not in locuses[i].neighbours:
                    locuses[i].neighbours.append((locuses[j], common_side))
                
                if (locuses[i], common_side) not in locuses[j].neighbours:
                    locuses[j].neighbours.append((locuses[i], common_side))
                
                edge = Segment2d(locuses[i].site, locuses[j].site)
                if edge not in triangulation:
                    triangulation.append(edge)
                
    return triangulation


def show_voronoi_delone(diagram : VoronoiDiagram, delone : list[Segment2d], bb):
    plt.grid()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    for locus in diagram.locuses:
        if len(locus.region.vertices > 0):
            plt.plot(locus.region.vertices[:, 0], locus.region.vertices[:, 1])
            plt.plot(locus.region.vertices[(-1,0), 0], locus.region.vertices[(-1,0), 1])
        plt.scatter([locus.site.x()], [locus.site.y()])

   
    
    for seg in delone:
        a = seg.a.as_array()
        b = seg.b.as_array()
        plt.plot([ a[0], b[0] ],[ a[1], b[1] ] , color='black')
    
    
    all_verts = { tuple(vert): (0.,0) for locus in diagram.locuses for vert in locus.region.vertices}
    for v in all_verts:    
        for locus in diagram.locuses:
            for locus_v in locus.region.vertices:            
                if np.all(np.isclose(locus_v, v, atol=EPS)):
                    all_verts[tuple(v)] = (float(np.linalg.norm(v - locus.site.as_array())), 
                                    all_verts[tuple(v)][1] + 1)
                    break

    all_verts = { k: v[0] for k, v in all_verts.items() if v[1] >= 3}
    
    # for center, radius in all_verts.items():
    #     circ = dis.Circle(center, radius, fill=False)
    #     plt.gca().add_patch(circ)
    
    plt.plot(bb[:,0], bb[:,1], color='blue')
    plt.plot(bb[(-1,0),0], bb[(-1,0),1], color='blue', linewidth=10**-2)


    plt.xlim( (bb[0][0] - 3, bb[1][0] + 3))
    plt.ylim( (bb[0][1] - 3, bb[2][1] + 3))

    
    plt.show()




def main():
    np.seterr(all='raise')
    np.set_printoptions(precision=2)
    points = gen_points(300, 200_000)
    
    
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
    
    # points = np.array([ [1, 1], [ 1, 2], [3, 1] ]) 
    
    # print(points.tolist())
    
    
    
    borderbox = borderbox_from_points(points)

    plt.grid()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')


    plt.plot(borderbox[:, 0], borderbox[:, 1])
    plt.scatter(points[:, 0], points[:, 1])
    
    plt.show()

    diagram = VoronoiDiagram.from_points(points, borderbox)

    diagram.plt_display(borderbox)

    delone = delone_triangulation(diagram)

    show_voronoi_delone(diagram, delone, borderbox)

    pass


if __name__ == '__main__':
    main()