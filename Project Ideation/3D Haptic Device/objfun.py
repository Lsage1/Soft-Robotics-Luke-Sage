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

def objfun(end_edge, first_end_vertex, edgeObjs, vertexObjs,
            qOld, uOld,
            freeIndex, dt, tol,
            massVector, massMatrix,
            EA, EI, GJ, Fg):

  q_new = qOld.copy()
  iter = 0
  error = 10 * tol

  while error > tol:
    # Reference frame
    # Compute a1_new, a2_new for each edge
    computeTimeParallel_OO(end_edge, first_end_vertex, edgeObjs, qOld, q_new) # Time parallel reference frame along the rod.
    # Reference twist
    ComputeTangentEdges(edgeObjs, True)


    refTwist_new = getRefTwist_OO(vertexObjs, edgeObjs, end_edge, first_end_vertex) # Reference twist vector of size nv


    quit()
    # Material frame
    theta = q_new[3::4]
    m1, m2 = computeMaterialDirectors(a1_new, a2_new, theta) # Material directors of size nv x 3

    # Computer elastic forces
    Fs, Js = getFs(q_new, EA, refLen)
    Fb, Jb = getFb(q_new, m1, m2, kappaBar, EI, voronoiRefLen)
    Ft, Jt = getFt(q_new, refTwist_new, twistBar, GJ, voronoiRefLen)

    Forces = Fs + Fb + Ft + Fg
    JForces = Js + Jb + Jt

    f = massVector / dt * ( (q_new - qOld) / dt - uOld ) - Forces
    J = massMatrix / dt**2 - JForces

    # Extract the free part
    f_free = f[freeIndex]
    J_free = J[np.ix_(freeIndex, freeIndex)]

    dq_free = np.linalg.solve(J_free, f_free) # J \ f

    q_new[freeIndex] -= dq_free
    error = np.sum(np.abs(f_free)) # Correction
    # Keep in mind that "error = np.sum(np.abs(dq_free))" is ok but tol should be computed based on length
    iter += 1

  uNew = (q_new - qOld) / dt
  return q_new, uNew, a1_new, a2_new