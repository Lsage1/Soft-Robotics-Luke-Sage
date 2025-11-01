

def getDirichlet(x_mid, y_mid, nv, q_guess, tol, maximum_iter, EI, EA, W, deltaL, free_index):

    dx = 10e-2 * deltaL # X perturbation value
    dy = 10e-2 * deltaL # y perturbation value
    dtheta = 10e-3 # Theta perturbation value

    q_original, error = staticSolver(q_guess, tol, maximum_iter, EI, EA, W, deltaL,free_index)
    if error < 0:
        print('Static Solver could not converge.')

    x_original = q_original[(nv+1)/2] # Get middle position

    return xc, yc, thetac, error