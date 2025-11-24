class VertexObj:
    def __init__(self, x,y):
        self.edges = []
        self.coords = (x,y)
        self.edgePairs = []

    def get_attached_verte(self):
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

    def get_other_vertex(self, vertex):
        if self.vertex1 == vertex:
            return self.vertex2
        else:
            return self.vertex1