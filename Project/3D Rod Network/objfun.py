import numpy as np
from computeTimeParallel import computeTimeParallel
from computeTangent import computeTangent
from getRefTwist import getRefTwist
from computeMaterialDirectors import computeMaterialDirectors
from getFs import getFs
from getFb import getFb
from getFt import getFt
from computeTimeParallel_OO import computeTimeParallel_OO
from ComputeTangentEdges import ComputeTangentEdges
from getRefTwist_OO import getRefTwist_OO
from computeMaterialDirectors_OO import computeMaterialDirectors_OO

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

    # Computer elastic forces
    Fs, Js = getFs(EA, vertexObjs, edgeObjs)
    #Fb, Jb = getFb(q_new, m1, m2, kappaBar, EI, voronoiRefLen)
    #Ft, Jt = getFt(q_new, refTwist_new, twistBar, GJ, voronoiRefLen)
    print("force: ", Fs)
    Forces = Fs + Fg#+ Fb + Ft + Fg
    JForces = Js #+ Jb + Jt

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

  u_new = (q_new - qOld) / dt
  return q_new, u_new