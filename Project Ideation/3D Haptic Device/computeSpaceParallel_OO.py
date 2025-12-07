import numpy as np
from computeTangent import computeTangent
from parallel_transport import parallel_transport

def computeSpaceParallel_OO(edge01, edge12):
    # we pass in two edges:
    #   edge01 - root edge that we will use to parallel transport from
    #   edge02 - edge we will parallel transport to

    # These tangents should already be unit vectors, but if not:
    t0 = edge01.tangent / np.linalg.norm(edge01.tangent)
    t1 = edge12.tangent / np.linalg.norm(edge12.tangent)

    edge12.u1 = parallel_transport(edge01.u1, t0, t1)
    edge12.u1 = edge12.u1 / np.linalg.norm(edge12.u1) # Ensure it is unit
    edge12.u2 = np.cross(t1, edge12.u1)
    edge12.u2 = edge12.u2 / np.linalg.norm(edge12.u2) # Ensure it is unit

    return