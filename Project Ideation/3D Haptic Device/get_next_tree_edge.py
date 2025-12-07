import numpy as np
from GeometryObjects import VertexObj
from GeometryObjects import EdgeObj

def get_next_tree_edge(prev_edge, joining_vertex):
    if not prev_edge.handled:
        raise ValueError("last edge wasn't handled yet")
    new_edge, doneFlag = joining_vertex.get_unhandled_edge()
    return new_edge, doneFlag