import numpy as np
from GeometryObjects import VertexObj
from GeometryObjects import EdgeObj

def get_next_tree_edge(prev_edge, joining_vertex):

    new_edge, found_edge = joining_vertex.get_unhandled_edge(prev_edge)

    return new_edge, found_edge