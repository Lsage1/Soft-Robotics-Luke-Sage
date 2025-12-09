import numpy as np
from computeTimeParallel import computeTimeParallel
from computeTangent import computeTangent
from getRefTwist import getRefTwist
from computeMaterialDirectors import computeMaterialDirectors
from getFs import getFs
from getFb import getFb
from getFt import getFt
from getFS_Junction import getFS_Junction

def objfun(qOld, uOld, qindex, jun_index, a1_old, a2_old,
           free_all_index,
           dt, tol,
           refTwist,
           massVector, massMatrix,
           EA, refLen,
           EI, GJ, voronoiRefLen,
           kappaBar, twistBar,
           f_ext_flat, nRods, tangent_old, nv # NV is the number of vertices in an edge
           ):
  print("objfun called")

  qNew = [qOld[i].copy() for i in range(nRods)]

  iter = 0
  error = 10 * tol

  ndof = 4 * nv - 1
  ne = nv - 1

  a1_new = np.zeros((nRods, ne, 3))
  a2_new = np.zeros((nRods, ne, 3))

  tangent = np.zeros((nRods, ne, 3))
  refTwist_new = np.zeros((nRods, nv))
  theta = np.zeros((nRods, ne))              # <- corrected size


  m1 = np.zeros((nRods, ne, 3))
  m2 = np.zeros((nRods, ne, 3))

  # Flatten Q Old
  q = [[], [], []]
  q_junction = qOld[0][:3]
  q[0] = qOld[0][3:]  # Remove the first 3 elements of q[1] because they will be shared with q[0] at the junction
  q[1] = qOld[1][3:]
  q[2] = qOld[2][3:]
  q_all_old = np.concatenate([q_junction, q[0], q[1], q[2]])

  # Flatten U Old
  u = [[], [], []]
  u_junction = uOld[0][:3]
  u[0] = uOld[0][3:]  # Remove the first 3 elements of q[1] because they will be shared with q[0] at the junction
  u[1] = uOld[1][3:]
  u[2] = uOld[2][3:]
  u_all_old = np.concatenate([u_junction, u[0], u[1], u[2]])

  while error > tol:

    #Flatten qNew
    q = [[], [], []]
    # REMOVE JUNCTION DOF FROM qOld matrix, PLACE IT IN q_junction
    q_junction = qNew[0][:3]
    q[0] = qNew[0][3:]  # Remove the first 3 elements of q[1] because they will be shared with q[0] at the junction
    q[1] = qNew[1][3:]
    q[2] = qNew[2][3:]
    q_all_new = np.concatenate([q_junction, q[0], q[1], q[2]])



    # ============ PER-ROD NEWTON WORK =============
    for c in range(nRods):

          # Reference frame transport
          a1_new[c], a2_new[c] = computeTimeParallel(a1_old[c], qOld[c], qNew[c])

          # Tangent and reference twist
          tangent[c] = computeTangent(qNew[c])
          refTwist_new[c] = getRefTwist(a1_new[c], tangent[c], refTwist[c])

          # ===== θ =====
          theta[c] = qNew[c][3::4]

          # Material directors
          m1[c], m2[c] = computeMaterialDirectors(a1_new[c], a2_new[c], theta[c])

    # Elastic forces

    Fs, Js = getFS_Junction(q_all_new, EA, refLen, jun_index, qindex, nRods, nv)

    #Fb, Jb = getFb(q_new[c], m1[c], m2[c], kappaBar[c], EI, voronoiRefLen[c])
    #Ft[c], Jt[c] = getFt(q_new[c], refTwist_new[c], twistBar[c], GJ, voronoiRefLen[c])

    Forces = Fs + f_ext_flat # + Fb[c] + Ft[c]
    JForces = Js # + Jb + Jt
    F = massVector / dt * ((q_all_new - q_all_old) / dt - u_all_old) - Forces
    J = massMatrix / dt**2 - JForces

    # Extract free components
    f_free = F[free_all_index]
    J_free = J[np.ix_(free_all_index, free_all_index)]

    dq_free = np.linalg.solve(J_free, f_free)
    q_all_new[free_all_index] -= dq_free

    # ---- Newton convergence ----


    error = np.sum(np.abs(f_free))
    print("iter: ", iter)
    iter += 1

    # UNFLATTEN A QNEW OR QOLD from q_all style vector
    qNew = []
    for c in range(nRods):
        qNew.append(np.concatenate([q_all_new[:3], q_all_new[qindex[c]]]))

    # final velocity
    uNew = [[], [], []]
    for c in range(nRods):
        uNew[c] = (qNew[c] - qOld[c]) / dt

  return qNew, uNew, a1_new, a2_new
