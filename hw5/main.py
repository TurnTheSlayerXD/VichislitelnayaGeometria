

import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon 

def display_figs(points : list, polygons : list, title:str = 'None'):
    
    
    color_points = np.array(['violet', 'blue', 'green', 'yellow',
                      'green',])
    colors = np.array(['black', 'red', 'yellow', 'yellow',
                      'green', 'blue', 'violet'])

    plt.axes()

    max_x = 0
    max_y = 0
    min_x = 100
    min_y = 100
    
    for poly in polygons:
        if len(poly) > 0:
            for p in poly[:, 0]:
                if max_x < p:
                    max_x = p
                if min_x > p:
                    min_x = p


    for poly in polygons:
        if len(poly) > 0:
            for p in poly[:, 1]:
                if max_y < p:
                    max_y = p
                if min_y > p:
                    min_y = p

    assert np.isscalar(max_x)
    assert np.isscalar(max_y)
    
    # plt.xlim(-min_x - 5, max_x + 5)
    # plt.ylim(-min_y - 5, max_y + 5)


    plt.grid()
    
    for i, poly in enumerate(polygons):
        if len(poly) > 0:
            t = Polygon(poly[:], alpha=0.5,
                            color=colors[i % len(colors)])
            plt.gca().add_patch(t)
        
    
    for i, batch in enumerate(points):
        if len(batch) > 0:
            plt.scatter(batch[:, 0], batch[:, 1], color=color_points[i % len(color_points)])
    plt.title(title)
        
    plt.show()


def display_points(points, segments, labels=['unknown']):
    colors = np.array(['black', 'red', 'orange', 'yellow',
                      'green', 'blue', 'violet'])
    plt.grid()
    plt.scatter(points[:, 0], points[:, 1], color=colors[0])
    
    if len(segments) > 0:
        plt.plot(segments[:, 0], segments[:, 1], color=colors[1])
        last = np.array([segments[-1], segments[0]])
        plt.plot(last[:, 0], last[:, 1], color=colors[1])
    
    np.clip
    plt.show()

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle= np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return angle

def find_start_point(points : np.ndarray):
    
    res = [sys.maxsize, sys.maxsize]
    ind = sys.maxsize
    for i, p in enumerate(points):
        if p[1] < res[1] or p[1] == res[1] and p[0] < res[0]:
            res = p
            ind = i 
    return ind

def sort_related(p0, points):
    base = np.array([10, 0])
    return sorted(points, key=lambda p: angle_between(base,p - p0))

def conv_grakham(points: np.ndarray):

    p0_ind = find_start_point(points)
    
    to_sort = np.concat([points[0 : p0_ind],
                        points[p0_ind+1:len(points)]])
    sorted_ps = sort_related(points[p0_ind], to_sort)
    p0 = points[p0_ind]

    stack = [p0, sorted_ps[0]]
    for i in range(1, len(sorted_ps)):
        assert len(stack) >= 2
        next_to_top = stack[-2]
        top = stack[-1]
        pi = sorted_ps[i]
        while np.cross(pi - top, next_to_top - top) < 0:
            stack.pop()
            assert len(stack) >= 2
            next_to_top = stack[-2]
            top = stack[-1]
        stack.append(pi)
    return np.array(stack)



def conv_jarvis(points):
    points = np.copy(points)
    def swap_and_pop(i):
        points[i], points[-1] = points[-1], points[i] 
        return points[:len(points) - 1] 
    p0 = points[0]
    j = 0
    for i in range(1, len(points)):
        if p0[1] > points[i][1] or (p0[1] == points[i][1] and p0[0] > points[i][1]):
            p0 = points[i]
            j = i
    conv = []
    conv.append(np.copy(points[j]))
    while True:
        p0 = conv[-1]        
        t = 0
        for k in range(1, len(points)):
            if np.cross(points[k] - p0, points[t] - p0) > 0:
                t = k
        conv.append(np.copy(points[t]))
        points = swap_and_pop(t)    
        if np.all(conv[-1] == conv[0]):
            break
    return np.array(conv)




def inside(p, edge_start, edge_end):
    return (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) - \
           (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0]) >= 0

def intersection(p1, p2, cp1, cp2):
    dx1 = p2[0] - p1[0]
    dy1 = p2[1] - p1[1]
    dx2 = cp2[0] - cp1[0]
    dy2 = cp2[1] - cp1[1]
    det = dx1 * dy2 - dy1 * dx2
    if det == 0:
        return None  
    s = ((cp1[0] - p1[0]) * dy2 - (cp1[1] - p1[1]) * dx2) / det
    return (p1[0] + s * dx1, p1[1] + s * dy1)

def sutherland_hodgman(subject_polygon, clip_polygon):
    output_list = subject_polygon
    for i in range(len(clip_polygon)):
        input_list = output_list
        output_list = []

        A = clip_polygon[i - 1]
        B = clip_polygon[i]

        for j in range(len(input_list)):
            P = input_list[j - 1]
            Q = input_list[j]

            if inside(Q, A, B):
                if not inside(P, A, B):
                    inter = intersection(P, Q, A, B)
                    if inter:
                        output_list.append(inter)
                output_list.append(Q)
            elif inside(P, A, B):
                inter = intersection(P, Q, A, B)
                if inter:
                    output_list.append(inter)
        
    return output_list


def point_in_polygon_ray(M, poly, ):
    x0, y0 = M
    cnt = 0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        if np.all(np.isclose(poly[i], M)) or  np.all(np.isclose(poly[(i+1) % n], M)):
            return True
        if ((y1 > y0) != (y2 > y0)):
            x_int = x1 + (y0 - y1) * (x2 - x1) / (y2 - y1)
            if x_int > x0:
                cnt += 1

    return cnt % 2 == 1



def gen_points(count, sqr_size):
    res = np.random.random_sample((count, 2)) * sqr_size

    return res

def get_area(polygon):
    s = 0.
    for i in range(len(polygon) - 1):
        s += polygon[i][0] * polygon[i + 1][1] - polygon[i + 1][0] * polygon[i][1]
    return s / 2

def get_perimetr(polygon):
    s = 0.
    for i in range(len(polygon) - 1):
        s += np.linalg.norm(polygon[i] - polygon[i+1])
    s += np.linalg.norm(polygon[0] - polygon[-1])
    
    return s


def task1():
    points_f = gen_points(20, 10**2)
    conv = conv_grakham(points_f)
 
    area = get_area(conv)
    perimetr = get_perimetr(conv)
 
    display_figs([points_f], [conv], f'grakham, area = {int(area)}, perimetr = {int(perimetr)}')
    
    
def task2():
    points_f = gen_points(20, 10**2)
    conv = conv_jarvis(points_f)
    
    area = get_area(conv)
    perimetr = get_perimetr(conv)
    display_figs([points_f], [conv], f'jarvis, area = {int(area)}, perimetr = {int(perimetr)}')


def task3():
    points_f = gen_points(10, 10**2)
    
    # points_f = np.array([ [0, 0], [10, 0], [10, 10], [0, 10]])
    conv = conv_grakham(points_f)
    
    points_s = gen_points(20, 10**2)
    # points_s = np.array([ [5, 0], [15, 0], [15, 15], [0, 10]])
    
    conv2 = conv_grakham(points_s)
    
    result = sutherland_hodgman(conv, conv2)
    result = np.array([list(p) for p in result])
    
    points_r = []
    for p in [*points_f, *points_s]:
        if point_in_polygon_ray(p, result):
            points_r.append(p)
            print(p)
            
    points_r = np.array(points_r)
    display_figs([points_f, points_s, points_r], [conv, conv2, result], 'intersection')


def main():
    warnings.filterwarnings('ignore')

    task1()
    task2()
    task3()

if __name__ == '__main__':
    main()
