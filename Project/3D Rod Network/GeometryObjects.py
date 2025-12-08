import numpy as np

class VertexObj:
    def __init__(self, x,y,z, index):
        self.children = []
        self.index = index
        self.edges = []
        self.rest_coords = (x,y,z)
        self.edgePairs = []
        self.rest_angles = []
        self.junction = False
        self.end = False
        self.voronoi_length = []
        self.coords = (x,y,z) # Current Coordinates
        self.coords0 = (x,y,z) # Previous iteration coordinates
        self.junction_handled = False
        self.dofs_fixed = [False, False, False]
        self.rest_kappa = [None, None]
        self.kappa = [None, None]
        self.ref_twist = []


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

    def get_unhandled_edge(self, prev_edge): # Return the next unhandled edge
        for edge in self.edges:
            if edge != prev_edge and edge.handled == False:

                #print("found an unhandled edge at vertex: ", self.coords, edge.get_other_vertex(self).coords)
                return edge, True # Found an unhandled Edge

        #print("Could not find an unhandled edge at vertex: ", self.coords)
        return None, False  # no unhandled edges

    def fix(self, dofs): # Call this to fix X,Y,Z Degrees of Freedom
        # Assign each DOF to avoid sharing list references
        for i in range(3):
            self.dofs_fixed[i] = bool(dofs[i])


class EdgeObj:
    def __init__(self, v1, v2, theta):
        self.children = []
        self.parent = None
        self.vertex1 = v1
        self.vertex2 = v2
        self.rest_length = 0
        self.tangent = None
        self.tangent0 = None
        self.rest_tangent = None
        self.u1 = None # 1st space parallel director
        self.u2 = None # 2nd space parallel director
        self.theta = theta
        self.a1 = None # Reference directors (before twist)
        self.a2 = None
        self.a1_old = None # Old reference directors - used to replace a1_old, a2_old
        self.a2_old = None
        self.m1 = None # 1st material director (after twist
        self.m2 = None # 2nd material director
        self.handled = False
        self.root = None # What edge was curvature and material directors calculated from?
        self.rest_twist = 0 # Initially set twist of each edge to be zero.
        self.twist = 0 # Initialize self.twist as equal to the ref_twist, which is zero
        self.twist_fixed = False

    def get_other_vertex(self, vertex):
        if self.vertex1 == vertex:
            return self.vertex2
        else:
            return self.vertex1

    def get_branch_number(self):
        parent = self.parent
        if parent is None:
            return None

        try:
            return parent.children.index(self)
        except ValueError("self is not in parent children"):
            # self is not in parent.children
            return None


