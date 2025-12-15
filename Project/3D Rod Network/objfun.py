import numpy as np
from computeTimeParallel import computeTimeParallel
from computeTangent import computeTangent
from getRefTwist import getRefTwist
from computeMaterialDirectors import computeMaterialDirectors
from getFs import getFs
from getFt import getFt
from computeTimeParallel_OO import computeTimeParallel_OO
from ComputeTangentEdges import ComputeTangentEdges
from getRefTwist_OO import getRefTwist_OO
from computeMaterialDirectors_OO import computeMaterialDirectors_OO
from getFb import getFb_OO_tree
from tree_traverse import tree_getKappa
from to_woven import to_woven
from to_woven import to_woven_j

np.set_printoptions(
    precision=7,       # number of decimals
    suppress=True,     # suppress scientific notation
    linewidth=240     # wrap lines nicely
)

def objfun(end_edge, first_end_vertex, edgeObjs, vertexObjs,
            qOld, uOld,
            freeIndex, dt, tol,
            massVector, massMatrix,
            EA, EI, GJ, Fg):

  q_new = qOld.copy()
  iter = 0
  error = 10 * tol
  getRefTwist_OO

  while error > tol:

    ################ UNPACK OBJECTS INTO QNEW ###################
    for vertex in vertexObjs:
        vertex.coords = q_new[vertex.index]

    # Insert twist DOFs
    for edge in edgeObjs:
        edge.theta = q_new[edge.theta_index]


    # Reference frame
    # Compute a1_new, a2_new for each edge
    computeTimeParallel_OO(end_edge, first_end_vertex, edgeObjs, qOld, q_new) # Time parallel reference frame along the rod.
    # Reference twist

    ComputeTangentEdges(edgeObjs, True)


    getRefTwist_OO(vertexObjs, edgeObjs, end_edge, first_end_vertex) # Reference twist vector of size nv


    for edge in edgeObjs:
        computeMaterialDirectors_OO(edge) # Material directors for each edge
    for edge in edgeObjs:
        edge.handled = False
    tree_getKappa(end_edge, first_end_vertex, vertexObjs, edgeObjs)

    # Computer elastic forces
    Fs, Js = getFs(EA, vertexObjs, edgeObjs)
    Fb, Jb = getFb_OO_tree(end_edge, first_end_vertex, vertexObjs, edgeObjs, EI, ndof_total=len(q_new))

    #Ft, Jt = getFt(q_new, refTwist_new, twistBar, edgeObjs, VertexObjs GJ, voronoiRefLen)

    Forces = Fs + Fg + Fb #+ Ft + Fg
    JForces = Js + Jb #+ Jt
    f = massVector / dt * ( (q_new - qOld) / dt - uOld ) - Forces
    J = massMatrix / dt**2 - JForces


    # Extract the free part
    f_free = f[freeIndex]
    J_free = J[np.ix_(freeIndex, freeIndex)]
    dq_free = np.linalg.solve(J_free, f_free) # J \ f

    q_new[freeIndex] -= dq_free

    error = np.sum(np.abs(f_free)) # Correction
    # Keep in mind that "error = np.sum(np.abs(dq_free))" is ok but tol should be computed based on length
    # REPACK INTO OBJECTS TO PASS TO NEXT ITERATION:
    # Update vertex coordinates
    for vertex in vertexObjs:
        vertex.coords = q_new[vertex.index]

    # Update edge twists
    for edge in edgeObjs:
        edge.theta = q_new[edge.theta_index]
    iter += 1
    print(q_new, iter)
  u_new = (q_new - qOld) / dt
  return q_new, u_new