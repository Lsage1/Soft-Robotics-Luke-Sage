import numpy as np
from computeKappa import computeKappa

def getKappa_OO(vertex, end_edge, edgeObjs, vertexObjs):
    vertex0 = vertex
    edge01 = end_edge

    vertex1 = edge01.get_other_vertex(vertex0)
    edge12 = vertex1.get_other_edge(edge01) ### EDGE 12 is considered the "Current" edge

    vertex2 = edge12.get_other_vertex(vertex1)

    active = True
    while active:  # Loop over edges, starting from an end point until a junction
        if vertex1.end or vertex1.junction:


        # Extract node positions from q
        node0 = vertex0.coords
        node1 = vertex1.coords
        node2 = vertex2.coords

        # Extract m1 and m2 for the current and previous edges
        m1e = edge01.m1  # m1 vector on c-1 th edge
        m2e = edge01.m2  # m2 vector on c-1 th edge
        m1f = edge12.m1  # m1 vector on c th edge
        m2f = edge12.m2  # m2 vector on c th edge

        # Compute local curvature at each node
        kappa_local = computeKappa(node0, node1, node2, m1e, m2e, m1f, m2f)

        edge12.kappa = kappa_local

    return kappa