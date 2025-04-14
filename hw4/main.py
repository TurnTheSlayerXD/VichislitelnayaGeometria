
from matplotlib.patches import Polygon
import numpy as np

import matplotlib.pyplot as plt

def display_points(points : list, segments, title = 'None'):
    
    color_points = np.array(['violet', 'blue', 'green', 'yellow',])
    
    colors = np.array(['black', 'red', 'orange', 'yellow',
                      'green', 'blue', 'violet'])
    plt.grid()
    
  
    for i, seg in enumerate(segments):
        plt.plot(seg[:, 0], seg[:, 1], color=colors[i % len(colors)], linewidth=3)
    
    
    for i, batch in enumerate(points):
        if len(batch) > 0:
            plt.scatter(batch[:, 0], batch[:, 1], color=color_points[ i % len(color_points)], linewidths=10)
    
   
    
    plt.title(title)
    
    plt.show()


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



def task1():
    segments = np.array([
    [[0, 0],   [5, 5]],    # 0
    [[0, 5],   [5, 0]],    # 1  → пересекается с 0
    [[1, 4],   [4, 1]],    # 2
    [[1, 1],   [4, 4]],    # 3  → пересекается с 2
    [[2, 0],   [2, 5]],    # 4
    [[0, 2],   [5, 2]],    # 5  → пересекается с 4
    [[3, -1],  [3, 6]],    # 6
    [[1, 3],   [6, 3]],    # 7  → пересекается с 6
    [[6, 0],   [6, 5]],    # 8  (без пересечений с предыдущими)
    [[0, 6],   [5, 6]],    # 9  (без пересечений с предыдущими)
])
   
    n = len(segments)
    intersections = []
    for i in range(n):
        for j in range(i + 1, n):
            pt = intersect_param(segments[i], segments[j])
            if pt is not None:
                intersections.append(pt)

    intersections = np.array(intersections)
    display_points([intersections], segments, title='Метод параметризации прямых')


def is_intersect_cross(seg1, seg2):
    A, B = seg1
    C, D = seg2
    res1 = np.cross(C - A, B - A) * np.cross(B - A, D - A) >= 0
    res2 = np.cross(A - D, C - D) * np.cross(C - D, B - D) >= 0
    return res1 and res2

def task2():
    segments = np.random.randint(0, 200, (10, 2, 2))

    n = len(segments)
    intersections = []
    for i in range(n):
        for j in range(i + 1, n):
            if is_intersect_cross(segments[i], segments[j]):
                intersections.append([segments[i], segments[j]])
                # plt.plot(segments[i][:, 0], segments[i][:, 1], color='green', linewidth=3)
                # plt.plot(segments[j][:, 0], segments[j][:, 1], color='green', linewidth=3)
                # plt.show()

    colors = np.array(['black', 'red', 'orange', 'yellow',
                      'green', 'blue', 'violet'])
    plt.grid()
    plt.title('Метод косых прямых')
    for seg in segments:
        plt.plot(seg[:, 0], seg[:, 1], color='black', linewidth=3)
   
    for seg1, seg2 in intersections:
        plt.plot(seg1[:, 0], seg1[:, 1], color='green', linewidth=3)
        plt.plot(seg2[:, 0], seg2[:, 1], color='green', linewidth=3)
   
    plt.show()
    


# type = 'start' | 'end' | 'intersect'

class Segment:
    def __init__(self, p1, p2):
        self.p1, self.p2  = (p1, p2) if p1[0] < p2[0] else (p2, p1)
        assert p1[0] != p2[0]        
        self.k = (p1[1] - p2[1]) / ( p1[0] - p2[0])
        self.b = p1[1] - self.k  * p1[0]
        
    def at(self, x) -> float:
        return self.k * x + self.b
    
    def __repr__(self):
        return f'[start={self.p1}, end={self.p2}]'    

class Event:
    def __init__(self, type : str, x_: float, seg: Segment | None = None, int_point : np.ndarray | None =  None):
        self.type = type    
        self.segment = seg
        self.x_ = x_
        self.int_point = int_point 
        
        self.seg1 : Segment | None = None
        self.seg2 : Segment | None = None


    def __lt__(self, rhs):
        return self.x_ < rhs.x_
    
    def __repr__(self) -> str:
        return f'[{self.type} x = {self.x_}]'
    
    

def intersect_segs(seg1 : Segment, seg2 : Segment):
    A, B = (seg1.p1, seg1.p2)
    C, D = (seg2.p1, seg2.p2)
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
        
        
        
def Bentley_Ottmann(segments):
    
    from heapq import heapify, heappush, heappop
    
    from bisect import bisect_left, bisect_right
    
    segments = [Segment(*seg) for seg in segments]    
    
    events : list[Event] = []
    
    status : list[Segment] = []
    
    
    for seg in segments:
        x_start, x_end = (seg.p1[0], seg.p2[0]) 
        heappush(events, Event('start', x_start, seg))
        heappush(events, Event('end', x_end, seg))

    intersections = []
    
    def add_intersection_event(seg1, seg2):
        int_p = intersect_segs(seg1, seg2)
        if int_p is not None:
            event = Event('intersect', int_p[0])
            event.seg1, event.seg2 = seg1, seg2
            event.int_point = int_p
            heappush(events, event)
            intersections.append(event.int_point)

    EPS = 10 ** -3
    while len(events) > 0:
        
        if len(events) > 500:
            print(events)
            assert False
        
        event = heappop(events)

        x = event.x_
        # print( [st.at(x) for st in status])
        # assert all(status[i].at(x) <= status[i+1].at(x) + EPS for i in range(len(status) - 1))
        if event.type == 'start':
            new_seg = event.segment
            y = new_seg.at(x)
            i = bisect_left(status, y + EPS, key=lambda seg : seg.at(x))
            if 0 < i:
                seg_left = status[i - 1]
                add_intersection_event(seg_left, new_seg)            

            if 0 <= i < len(status):
                seg_right = status[i]
                add_intersection_event(seg_right, new_seg)            

            status.insert(i, new_seg)
            print( [st.at(x) for st in status])
           
            # assert all(status[i].at(x) < status[i+1].at(x) or np.isclose(status[i].at(x), status[i+1].at(x))
            #            for i in range(len(status) - 1))
            
                
        elif event.type == 'end':
            cur_seg = event.segment
            y = cur_seg.p2[1]
            i = bisect_left(status, y + EPS, key=lambda seg : seg.at(x))

            if i == len(status):
                i -= 1
                
            if not np.isclose(status[i].at(x), y, atol=10**-3):
                i -= 1
            print( y, [st.at(x) for st in status])
            print(y, status[i].at(x))
            
            # assert np.isclose(status[i].at(x), y, atol=10**-3)
            try: 
                status.pop(i)
            except IndexError:
                assert False
            
            if i > 0 and i < len(status):
                neigh_left = status[i - 1]
                neigh_right = status[i]
                add_intersection_event(neigh_left, neigh_right)
            assert all(status[i].at(x) < status[i+1].at(x) or np.isclose(status[i].at(x), status[i+1].at(x))
                       for i in range(len(status) - 1))
                
        else:
            y = event.int_point[1]
            i = bisect_right(status, y + EPS, key=lambda seg: seg.at(x))
            
            print(status[i-1].at(x), status[i-2].at(x), y)

            # assert np.isclose(status[i-1].at(x), status[i-2].at(x), atol=10**-3)
            
            status[i-1], status[i-2] = status[i-2], status[i-1]
            if i < len(status):            
                add_intersection_event(status[i-1], status[i])
            if i-3 > -1:
                add_intersection_event(status[i-2], status[i-3])
                
            print( [st.at(x) for st in status])
            # assert all(status[i].at(x) < status[i+1].at(x) or np.isclose(status[i].at(x), status[i+1].at(x))
            #            for i in range(len(status) - 1))
                
    return intersections

def task3():
    
    segments = np.random.randint(10, 100, (5, 2, 2))
    
    segments = np.array([[[1, 1], [2, 5]],
                        [[2, 7], [5, -1]],
                        [[3, 4], [0, 5]],
                        [[3, 1], [2, 6]],
                        [[4, 8], [5, 0]]])
    
    res = Bentley_Ottmann(segments)
    

    print(f'res={np.array(res)}')
    display_points([np.array(res)], segments, title="Bentley_Ottmann")
    
        
    pass



def point_in_polygon_angle(M, poly):
    EPS=1e-4
    total_angle = 0.
    n = len(poly)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i+1) % n]
        cross = np.cross(p1 - M, p2 - M)
        dot = np.dot(p1 - M, p2 - M)
        if abs(cross) < EPS and dot <= 0:
            return True
        angle = np.atan2(cross, dot)
        total_angle += angle

    if np.isclose(total_angle, 2 * np.pi):
        return True
    return False
    
    
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




def display_figs(points : list, polygons : list, title:str = 'None'):
    color_points = np.array(['violet', 'blue', 'green', 'yellow',
                      'green',])
    colors = np.array(['black', 'red', 'yellow', 'yellow',
                      'green', 'blue', 'violet'])
    plt.grid()
    
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

    
    plt.xlim(-min_x - 5, max_x + 5)
    plt.ylim(-min_y - 5, max_y + 5)

    
    for i, poly in enumerate(polygons):
        if len(poly) > 0:
            t = Polygon(poly[:], alpha=0.5,
                            color=colors[i % len(colors)])
            plt.gca().add_patch(t)
        
    
    for i, batch in enumerate(points):
        if len(batch) > 0:
            plt.scatter(batch[:, 0], batch[:, 1], color=color_points[ i % len(color_points)])
    plt.title(title)
        
    plt.show()


def task4():
    
    points = gen_points(20, 20)
    
    fig = conv_jarvis(points[:10])
    
    in_poly = []
    out_poly = []
    for p in points:
        if point_in_polygon_angle(p, fig):
            in_poly.append(p)
        else:
            out_poly.append(p)

    display_figs([np.array(in_poly), np.array(out_poly) ], [fig], 'Угловой метод')
                


def task5():
    
    points = gen_points(20, 20)
    
    fig = conv_jarvis(points[:10])
    
    in_poly = []
    out_poly = []
    for p in points:
        if point_in_polygon_ray(p, fig):
            in_poly.append(p)
        else:
            out_poly.append(p)

    display_figs([np.array(in_poly), np.array(out_poly) ], [fig], 'Лучевой метод')
                


def main():
    task1()
    task2()
    task3()
    task4()
    task5()
    
    

if __name__ == '__main__':
    main()