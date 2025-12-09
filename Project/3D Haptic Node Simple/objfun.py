import numpy as np
from computeTimeParallel import computeTimeParallel
from computeTangent import computeTangent
from getRefTwist import getRefTwist
from computeMaterialDirectors import computeMaterialDirectors
from getFs import getFs
from getFb import getFb
from getFt import getFt
from getFS_Junction import getFS_Junction

def objfun(qOld, uOld, a1_old, a2_old,
           freeIndex,
           dt, tol,
           refTwist,
           massVector, massMatrix,
           EA, refLen, refLenJunction,
           EI, GJ, voronoiRefLen,
           kappaBar, twistBar,
           Fg, nRods, tangent_old, nv):
  print("objfun called")
  q_new = [q.copy() for q in qOld]
  iter = 0
  error = 10 * tol

  ndof = 4 * nv - 1
  ne = nv - 1

  a1_new = np.zeros((nRods, ne, 3))
  a2_new = np.zeros((nRods, ne, 3))

  tangent = np.zeros((nRods, ne, 3))
  refTwist_new = np.zeros((nRods, nv))
  theta = np.zeros((nRods, ne))              # <- corrected size

  Fs = np.zeros((nRods, ndof))
  Js = np.zeros((nRods, ndof, ndof))
  m1 = np.zeros((nRods, ne, 3))
  m2 = np.zeros((nRods, ne, 3))
  Fb = np.zeros((nRods, ndof))
  Jb = np.zeros((nRods, ndof, ndof))
  Ft = np.zeros((nRods, ndof))
  Jt = np.zeros((nRods, ndof, ndof))


  JForces = np.zeros((nRods, ndof, ndof))



  f_free = np.zeros((nRods, len(freeIndex[0])))
  J_free = np.zeros((nRods, len(freeIndex[0]), len(freeIndex[0])))
  dq_free = np.zeros((nRods, len(freeIndex[0])))

  uNew = np.zeros((nRods, ndof))

  while error > tol:

    # ---- Junction compatibility constraint ----
    junction_pos = np.mean([q_new[c][0:3] for c in range(nRods)], axis=0)
    for c in range(nRods):
       q_new[c][0:3] = junction_pos

    # ============ PER-ROD NEWTON WORK =============
    for c in range(nRods):

          # Reference frame transport
          a1_new[c], a2_new[c] = computeTimeParallel(a1_old[c], qOld[c], q_new[c])

          # Tangent and reference twist
          tangent[c] = computeTangent(q_new[c])
          refTwist_new[c] = getRefTwist(a1_new[c], tangent[c], refTwist[c])

          # ===== θ =====
          theta[c] = q_new[c][3::4]

          # Material directors
          m1[c], m2[c] = computeMaterialDirectors(a1_new[c], a2_new[c], theta[c])

    # Elastic forces
    Fs, Js = getFS_Junction(q_new, EA, refLen, refLenJunction)
    #Fb, Jb = getFb(q_new[c], m1[c], m2[c], kappaBar[c], EI, voronoiRefLen[c])
    #Ft[c], Jt[c] = getFt(q_new[c], refTwist_new[c], twistBar[c], GJ, voronoiRefLen[c])

    Forces = Fs + Fg # + Fb[c] + Ft[c]
    JForces = Js + Jb + Jt
    F = massVector / dt * ((q_new - qOld) / dt - uOld) - Forces
    J = massMatrix / dt**2 - JForces

    # Extract free components
    f_free[c] = F[c][freeIndex[c]]
    J_free[c] = J[c][np.ix_(freeIndex[c], freeIndex[c])]

    dq_free[c] = np.linalg.solve(J_free[c], f_free[c])
    q_new[c][freeIndex[c]] -= dq_free[c]

    # ---- Newton convergence ----


    error = np.max(np.abs(f_free))
    print("iter: ", iter)
    iter += 1
    # final velocity
    for c in range(nRods):
        uNew[c] = (q_new[c] - qOld[c]) / dt

    return q_new, uNew, a1_new, a2_new
