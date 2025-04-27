
import numpy as np

EPS = 1e-5

def inside(p, edge_start, edge_end):
    val = (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) - \
           (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0])
    
    return val > 0 or np.isclose(val , 0, atol = EPS)


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


def sutherland_hodgman(subject_polygon: np.ndarray, clip_polygon: np.ndarray) -> np.ndarray:
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

    return np.array(output_list)
