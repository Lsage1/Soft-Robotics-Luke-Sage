import numpy as np
from computeTimeParallel import computeTimeParallel
from computeTangent import computeTangent
from getRefTwist import getRefTwist
from computeMaterialDirectors import computeMaterialDirectors
from getFs import getFs
from getFb import getFb
from getFt import getFt

def objfun_test(qOld, uOld, a1_old, a2_old,
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
  theta = np.zeros((nRods, ne))

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

  uNew = np.zeros((nRods, ndof))

  # Precompute stacked sizes
  ndof_stack = nRods * ndof

  # Build the constraint matrix C that enforces equality of first vertex translations
  # We choose rod 0 as reference; for each rod c=1..nRods-1 add constraints:
  # q_ref[0:3] - q_c[0:3] = 0  (3 scalar rows per rod)
  nc = 3 * (nRods - 1) if nRods > 1 else 0
  if nc > 0:
    C_full = np.zeros((nc, ndof_stack))
    row = 0
    for c in range(1, nRods):
      # index of translational DOFs of first vertex in stacked vector
      idx_ref = 0 * ndof + 0           # rod0 start + translational offset (0)
      idx_c   = c * ndof + 0           # rod c start + translational offset
      for k in range(3):               # x,y,z
        C_full[row, idx_ref + k] = 1.0
        C_full[row, idx_c   + k] = -1.0
        row += 1
  else:
    C_full = np.zeros((0, ndof_stack))  # no constraints if single rod

  # Newton loop (global augmented solve)
  while error > tol:

    # --- per-rod evaluation (use current q_new) ---
    for c in range(nRods):

      # Reference frame transport (use qOld and q_new for the transport as before)
      a1_new[c], a2_new[c] = computeTimeParallel(a1_old[c], qOld[c], q_new[c])

      # Tangent and reference twist
      tangent[c] = computeTangent(q_new[c])
      refTwist_new[c] = getRefTwist(a1_new[c], tangent[c], refTwist[c])

      # theta and material directors
      theta[c] = q_new[c][3::4]
      m1[c], m2[c] = computeMaterialDirectors(a1_new[c], a2_new[c], theta[c])

      # Elastic forces and Jacobians
      Fs[c], Js[c] = getFs(q_new[c], EA, refLen[c])
      Fb[c], Jb[c] = getFb(q_new[c], m1[c], m2[c], kappaBar[c], EI, voronoiRefLen[c])
      Ft[c], Jt[c] = getFt(q_new[c], refTwist_new[c], twistBar[c], GJ, voronoiRefLen[c])

      Forces[c] = Fs[c] + Fb[c] + Ft[c] + Fg[c]
      JForces[c] = Js[c] + Jb[c] + Jt[c]

      # rod-local residual and jacobian (same definitions as you had)
      f[c] = massVector / dt * ((q_new[c] - qOld[c]) / dt - uOld[c]) - Forces[c]
      J[c] = massMatrix / dt**2 - JForces[c]

    # --- assemble stacked global residual and jacobian ---
    f_stack = np.zeros(ndof_stack)
    J_stack = np.zeros((ndof_stack, ndof_stack))
    for c in range(nRods):
      start = c * ndof
      f_stack[start:start + ndof] = f[c]
      J_stack[np.ix_(range(start, start + ndof), range(start, start + ndof))] = J[c]

    # --- build free-global index list from freeIndex per rod ---
    free_global_list = []
    for c in range(nRods):
      # freeIndex[c] is assumed an iterable of local indices
      free_global_list.extend([c * ndof + i for i in freeIndex[c]])
    free_global = np.array(free_global_list, dtype=int)
    n_free = free_global.size

    # if no free dofs, break
    if n_free == 0:
      break

    # restrict J,f to free DOFs
    J_ff = J_stack[np.ix_(free_global, free_global)]
    f_f  = f_stack[free_global]

    # restrict constraints to free unknowns
    if nc > 0:
      C_f = C_full[:, free_global]          # shape (nc, n_free)
      # build augmented system
      # [ J_ff   C_f^T ] [ dq_free ] = [ f_f ]
      # [ C_f    0    ] [  lambda ]   [  0  ]
      top = np.concatenate([J_ff, C_f.T], axis=1)             # (n_free, n_free+nc)
      bottom = np.concatenate([C_f, np.zeros((nc, nc))], axis=1)  # (nc, n_free+nc)
      Kaug = np.vstack([top, bottom])                         # (n_free+nc, n_free+nc)
      rhs = np.concatenate([f_f, np.zeros(nc)])
    else:
      # no constraints -> simple reduced system
      Kaug = J_ff
      rhs = f_f

    # solve for dq_free (and lambda if present)
    try:
      dx = np.linalg.solve(Kaug, rhs)
    except np.linalg.LinAlgError:
      # fallback to a damped solve or pseudo-inverse; here we do pinv
      dx = np.linalg.pinv(Kaug).dot(rhs)

    if nc > 0:
      dq_free = dx[:n_free]
      # lambda_vals = dx[n_free:]   # multipliers (not used further here)
    else:
      dq_free = dx

    # scatter dq_free into full stacked delta and update q_new
    dq_stack = np.zeros(ndof_stack)
    dq_stack[free_global] = dq_free

    for c in range(nRods):
      start = c * ndof
      dq_local = dq_stack[start:start + ndof]
      q_new[c] -= dq_local

    # compute convergence metric (use max abs residual on free components)
    # update residuals f for new q_new next iteration; here we use the last assembled f_f norm
    error = np.max(np.abs(f_f))
    print("iter:", iter, "error:", error)
    iter += 1

  # final velocity
  for c in range(nRods):
    uNew[c] = (q_new[c] - qOld[c]) / dt

  return q_new, uNew, a1_new, a2_new
