import numpy as np
from computeKappa import computeKappa

def getKappa_Junction(q, m1, m2):
    # Take in q: assumes that q is in the form [[x1,y1,z1,t1...], [x1,y1,z1,t1...], [x1,y1,z1,t1...]]
    # This makes it easier to take the second vertex in each list. these vertices connect to the junction vertex
    print("q", q)
    print("m1", m1)

    kappa_junction = [None, None, None]  # Initialize kappa array


    # Extract node positions from Branch 0, 1
    node0 = q[0][4:7]
    node1 = q[0][:3]
    node2 = q[1][4:7]

    # Extract m1 and m2 for the current and previous edges
    m1e = m1[0][0]
    m2e = m2[0][0]
    m1f = m1[1][0]
    m2f = m2[1][0]
    # Compute local curvature between branch 0,1
    kappa_junction[0] = computeKappa(node0, node1, node2, m1e, m2e, m1f, m2f)
    # Extract node positions from Branch 1, 2
    node0 = q[1][4:7]
    node1 = q[0][:3]
    node2 = q[2][4:7]

    # Extract m1 and m2 for the current and previous edges
    m1e = m1[1][0]
    m2e = m2[1][0]
    m1f = m1[2][0]
    m2f = m2[2][0]

    # Compute local curvature between branch 1,2
    kappa_junction[1] = computeKappa(node0, node1, node2, m1e, m2e, m1f, m2f)

    # Extract node positions from Branch 2,0
    node0 = q[2][4:7]
    node1 = q[0][:3]
    node2 = q[0][4:7]

    # Extract m1 and m2 for the current and previous edges
    m1e = m1[2][0]
    m2e = m2[2][0]
    m1f = m1[0][0]
    m2f = m2[0][0]
    kappa_junction[2] = computeKappa(node0, node1, node2, m1e, m2e, m1f, m2f)

    print("kappa_junction", kappa_junction)
    return kappa_junction