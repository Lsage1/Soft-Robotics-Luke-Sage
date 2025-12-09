import numpy as np
from computeTimeParallel import computeTimeParallel
from computeTangent import computeTangent
from getRefTwist import getRefTwist
from computeMaterialDirectors import computeMaterialDirectors
from getFs import getFs
from getFb import getFb
from getFt import getFt

def objfun(qOld, uOld, a1_old, a2_old,
           freeIndex,
           dt, tol,
           refTwist,
           massVector, massMatrix,
           EA, refLen,
           EI, GJ, voronoiRefLen,
           kappaBar, twistBar,
           Fg, nRods, tangent_old, nv):

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


  Forces = np.zeros((nRods, ndof))
  JForces = np.zeros((nRods, ndof, ndof))

  f = np.zeros((nRods, ndof))
  J = np.zeros((nRods, ndof, ndof))

  f_free = np.zeros((nRods, len(freeIndex[0])))
  J_free = np.zeros((nRods, len(freeIndex[0]), len(freeIndex[0])))
  dq_free = np.zeros((nRods, len(freeIndex[0])))

  uNew = np.zeros((nRods, ndof))

  while error > tol:

    # ---- Junction compatibility constraint ----
    # junction_pos = np.mean([q_new[c][0:3] for c in range(nRods)], axis=0)
    # for c in range(nRods):
    #   q_new[c][0:3] = junction_pos

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
      Fs[c], Js[c] = getFs(q_new[c], EA, refLen[c])
      Fb[c], Jb[c] = getFb(q_new[c], m1[c], m2[c], kappaBar[c], EI, voronoiRefLen[c])
      Ft[c], Jt[c] = getFt(q_new[c], refTwist_new[c], twistBar[c], GJ, voronoiRefLen[c])

      Forces[c] = Fs[c] + Fb[c] + Ft[c] + Fg[c]
      print("F: ", Forces[c])
      JForces[c] = Js[c] + Jb[c] + Jt[c]
      f[c] = massVector / dt * ((q_new[c] - qOld[c]) / dt - uOld[c]) - Forces[c]
      J[c] = massMatrix / dt**2 - JForces[c]
      # Extract free components
      f_free[c] = f[c][freeIndex[c]]
      print("f_free: ", f_free[c])
      J_free[c] = J[c][np.ix_(freeIndex[c], freeIndex[c])]

      dq_free[c] = np.linalg.solve(J_free[c], f_free[c])
      q_new[c][freeIndex[c]] -= dq_free[c]

    # ---- Newton convergence ----
    error = np.max(np.abs(f_free))
    print(iter, error)
    iter += 1
  # final velocity
  for c in range(nRods):
    uNew[c] = (q_new[c] - qOld[c]) / dt

  return q_new, uNew, a1_new, a2_new
