import numpy as np
from computeKappa import computeKappa

def getKappa_OO(node0, node1, node2, edge01, edge12):
    # Extract node positions from vertex objects


    # Extract m1 and m2 for the current and previous edges
    m1e = edge01.m1  # m1 vector on previous  edge
    m2e = edge01.m2  # m2 vector on previous edge
    m1f = edge12.m1  # m1 vector on current edge
    m2f = edge12.m2  # m2 vector on current edge

    # Compute local curvature at each node
    kappa_local = computeKappa(node0, node1, node2, m1e, m2e, m1f, m2f)

    # Store the curvature values
    # vertex1.kappa = kappa_local


    return kappa_local