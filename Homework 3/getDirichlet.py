

def getDirichlet(x_mid, y_mid, q_guess, tol, maximum_iter, EI, EA, W, deltaL, free_index):

    dx = 10e-2 * deltaL # X perturbation value
    dy = 10e-2 * deltaL # y perturbation value
    dtheta = 10e-3 # Theta perturbation value

    q_predicted, error = staticSolver(q_guess, tol, maximum_iter, EI, EA, W, deltaL,free_index)
    if error < 0:
        print('Static Solver could not converge.')


    return xc, yc, thetac, error