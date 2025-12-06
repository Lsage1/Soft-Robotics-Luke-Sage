import numpy as np

class VertexObj:
    def __init__(self, x,y,z, index):
        self.index = index
        self.edges = []
        self.rest_coords = (x,y,z)
        self.edgePairs = []
        self.rest_angles = []
        self.junction = False
        self.end = False
        self.voronoi_length = []
        self.coords = (x,y,z)


    def get_attached_vertex(self):
        attached_verts = []
        for edge in self.edges:
            attached_verts.append(edge.get_other_vertex(self))

    def add_edge(self, new_edge):
        for edge in self.edges:
            self.edgePairs.append((edge, new_edge))
        self.edges.append(new_edge)

    def get_other_edge(self, edge):
        if len(self.edges) != 2:
            raise ValueError("Can't get other edge from this vertex, Too many or too few edges")
        else:
            if self.edges[0] == edge:
                return self.edges[1]
            else:
                return self.edges[0]






class EdgeObj:
    def __init__(self, v1, v2, theta):
        self.vertex1 = v1
        self.vertex2 = v2
        self.rest_length = 0
        self.tangent = []
        self.u1 = [] # 1st space parallel director
        self.u2 = [] # 2nd space parallel director
        self.theta = theta
        self.m1 = [] # 1st material director
        self.m2 = [] # 2nd material director

    def get_other_vertex(self, vertex):
        if self.vertex1 == vertex:
            return self.vertex2
        else:
            return self.vertex1

