import numpy as np
from computeKappa import computeKappa

def getKappa_OO(vertex0, vertex1, vertex2, edge01, edge12):



    for c in range(2, nv):  # Loop over edges (from second to last)

        # Extract node positions from q
        node0 = np.array(vertex0.coords)
        node1 = np.array(vertex1.coords)
        node2 = np.array(vertex2.coords)

        # Extract m1 and m2 for the current and previous edges
        m1e = edge01.m1  # m1 vector on c-1 th edge
        # Another option is m1e = np.squeeze(np.array(m1[c-2, :]))
        m2e = edge01.m2  # m2 vector on c-1 th edge
        m1f = edge12.m1  # m1 vector on c th edge
        m2f = edge12.m2  # m2 vector on c th edge

        # Compute local curvature at each node
        kappa_local = computeKappa(node0, node1, node2, m1e, m2e, m1f, m2f)

        # Store the curvature values
        edge12.m1 = kappa_local[0]
        edge12.m2 = kappa_local[1]

    return