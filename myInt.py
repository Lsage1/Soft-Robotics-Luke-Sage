# Integrator
import numpy as np
from getForceJacobian import getForceJacobian


def myInt(t_new, x_old, u_old, free_DOF, stiffness_matrix, index_matrix, m, dt, l_k):

  # Guess solution for x_new
  x_new = x_old.copy() # Start with a copy of x_old as the initial guess

  eps = 1.0e-6 # tolerance
  err = eps * 100  # initialize to a large value

  # Newton-Raphson method
  while err > eps:
    f, J = getForceJacobian(x_new, x_old, u_old, stiffness_matrix, index_matrix, m, dt, l_k)

    # Extract free DOFs
    J_free = J[np.ix_(free_DOF, free_DOF)]
    f_free = f[free_DOF]

    # Solve for deltaX of free DOFs
    deltaX_free = np.linalg.solve(J_free, f_free)

    # Construct full deltaX vector
    deltaX = np.zeros_like(x_new)
    deltaX[free_DOF] = deltaX_free

    # Update x_new
    x_new = x_new - deltaX

    # Calculate error based on the force in free DOFs
    err = np.linalg.norm(f_free)

  u_new = (x_new - x_old) / dt

  return x_new, u_new