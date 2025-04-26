    def get_cur_vertex(self) -> Point2d:
        return Point2d(self.vertices[self.cur_vertex_ind])

    def get_cur_edge(self):
        return Segment2d(self.get_cur_vertex(),
                         Point2d(self.vertices[(self.cur_vertex_ind + 1) % len(self.vertices)]))

    @staticmethod
    def edge_case_num(lhs: Segment2d, rhs: Segment2d) -> str:
        first_look_at = lhs.looks_at(rhs)
        second_look_at = rhs.looks_at(lhs)

        if first_look_at and second_look_at:
            return 'kBothLooks'
        elif first_look_at:
            return 'kFirstLooksAtSecond'
        elif second_look_at:
            return 'kSecondLooksAtFirst'
        else:
            return 'kBothNotLooks'

    @staticmethod
    def which_edge_is_inside(first_edge: Segment2d, second_edge: Segment2d) -> str:

        first_second_inside = np.cross(second_edge.as_vector(),
                                       np.array([first_edge.b.x() - second_edge.a.x(),
                                                 first_edge.b.y() - second_edge.a.y()]))
        second_first_inside = np.cross(first_edge.as_vector(),
                                       np.array([second_edge.b.x() - first_edge.a.x(),
                                                 second_edge.b.y() - first_edge.a.y()]))
        if first_second_inside < 0:
            return 'kSecondEdge'
        elif second_first_inside < 0:
            return 'kFirstEdge'
        else:
            return 'Unknown'

    def addNewVertex(self, vertex: Point2d):
        self.vertices = np.append(self.vertices, vertex.as_array())
        self.cur_vertex_ind = (self.cur_vertex_ind + 1) % len(self.vertices)

    @staticmethod
    def move_on_edges(first_edge: Segment2d, second_edge: Segment2d, result: 'Polygon'):
        now_inside = Polygon.which_edge_is_inside(first_edge, second_edge)
        case_num = Polygon.edge_case_num(first_edge, second_edge)

        which_edge_is_moving = 0

        if case_num == 'kBothLooks':
            if now_inside == 'kFirstEdge':
                which_edge_is_moving = 'kSecondEdge'
            else:
                which_edge_is_moving = 'kFirstEdge'
        elif case_num == 'kFirstLooksAtSecond':
            which_edge_is_moving = 'kFirstEdge'
        elif case_num == 'kSecondLooksAtFirst':
            which_edge_is_moving = 'kSecondEdge'
        elif case_num == 'kBothNotLooks':
            if now_inside == 'kFirstEdge':
                which_edge_is_moving = 'kSecondEdge'
            else:
                which_edge_is_moving = 'kFirstEdge'

        if len(result.vertices) != 0 and (case_num == 'kFirstLooksAtSecond'
                                          or case_num == 'kSecondLooksAtFirst'):
            if now_inside == 'kFirstEdge':
                vertex_to_add = first_edge.b
            elif now_inside == 'kSecondEdge':
                vertex_to_add = second_edge.b
            else:
                if case_num == 'kFirstLooksAtSecond':
                    vertex_to_add = first_edge.b
                else:
                    vertex_to_add = second_edge.b

            if np.all(vertex_to_add.as_array() != result.get_cur_vertex().as_array()):
                result.addNewVertex(vertex_to_add)

        return which_edge_is_moving
    
    
def intersect_with_poly(self, rhs: 'Polygon') -> 'Polygon':

        max_iter = 2 * len(self.vertices + len(rhs.vertices))

        no_intersection = True

        moving_edge = 'Unknown'

        result = Polygon()

        for i in range(max_iter):
            cur_fp_edge = self.get_cur_edge()
            cur_sp_edge = rhs.get_cur_edge()

            inter_point = cur_fp_edge.get_point_intersection(cur_sp_edge)
            if type(inter_point) is not PosInfPoint2d:
                if len(result.vertices) == 0:
                    no_intersection = False
                    result.addNewVertex(inter_point)
                elif np.all(inter_point.as_array() != result.get_cur_vertex().as_array()):
                    if np.all(inter_point.as_array() == result.vertices[0]):
                        break
                    else:
                        result.addNewVertex(inter_point)

            moving_edge = Polygon.move_on_edges(
                cur_fp_edge, cur_sp_edge, result)
            if moving_edge == 'kFirstEdge':
                self.cur_vertex_ind = (
                    self.cur_vertex_ind + 1) % len(self.vertices)
            else:
                rhs.cur_vertex_ind = (
                    rhs.cur_vertex_ind + 1) % len(rhs.vertices)

        return result