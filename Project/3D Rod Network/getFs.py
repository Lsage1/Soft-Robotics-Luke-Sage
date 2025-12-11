import numpy as np
from gradEs_hessEs import gradEs_hessEs

import numpy as np
from gradEs_hessEs import gradEs_hessEs

def getFs(EA, vertexObjs, edgeObjs):
    """
    Compute stretching forces and Jacobian using OO data structures.
    EA is a scalar (same EA for all edges).
    """
    ndof = len(vertexObjs) * 3 + len(edgeObjs)     # translation DOFs + twist DOFs
    Fs = np.zeros(ndof)
    Js = np.zeros((ndof, ndof))

    for edge in edgeObjs:

        # Vertex positions
        xa = np.array(edge.vertex1.coords)
        xb = np.array(edge.vertex2.coords)

        # DOF indices (already stored)
        ia = edge.vertex1.index   # [i, i+1, i+2]
        ib = edge.vertex2.index   # [j, j+1, j+2]

        # Combined list of 6 DOFs
        ind = np.array([ia[0], ia[1], ia[2],
                        ib[0], ib[1], ib[2]])

        # Rest length
        dL = edge.rest_length

        # Compute element force and Jacobian
        dF, dJ = gradEs_hessEs(xa, xb, dL, EA)

        # Assemble
        Fs[ind] -= dF
        Js[np.ix_(ind, ind)] -= dJ

    return Fs, Js


