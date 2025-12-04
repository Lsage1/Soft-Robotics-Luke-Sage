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


    def get_attached_vertex(self):
        attached_verts = []
        for edge in self.edges:
            attached_verts.append(edge.get_other_vertex(self))

    def add_edge(self, new_edge):
        for edge in self.edges:
            self.edgePairs.append((edge, new_edge))
        self.edges.append(new_edge)






class EdgeObj:
    def __init__(self, v1, v2):
        self.vertex1 = v1
        self.vertex2 = v2
        self.rest_length = 0

    def get_other_vertex(self, vertex):
        if self.vertex1 == vertex:
            return self.vertex2
        else:
            return self.vertex1

