from staticSolver import staticSolver
import matplotlib.pyplot as plt
import numpy as np

def getDirichlet(xc, yc, thetac, x_mid, y_mid, nv,
                 q0, # guess beam positions from last time step.
                 tol, maximum_iter, EI, EA, W, deltaL, free_index):

    # Set the boundary conditions to a guess from the last frames.
    q0[nv * 2 - 2] = xc  # x_n
    q0[nv * 2 - 1] = yc  # y_n
    q0[nv * 2 - 4] = xc - deltaL * np.cos(thetac)  # X n-1
    q0[nv * 2 - 3] = yc - deltaL * np.sin(thetac)  # Y n-1

    dx = 10e-2 * deltaL # X perturbation value
    dy = 10e-2 * deltaL # y perturbation value
    dtheta = 10e-3 # Theta perturbation value


    # Get reference position of all nodes given guess end conditions xc, yc, thetac.
    print(x_mid, y_mid)
    q_original, error = staticSolver(q0, tol, maximum_iter, EI, EA, W, deltaL,free_index)
    if error < 0:
        print('Static Solver could not converge.')

    print("beam middle position: ", q_original[nv+1], q_original[nv])


    # Get position of all nodes if X is perturbed based on guess
    q_p_x, error = staticSolver(q0, tol, maximum_iter, EI, EA, W, deltaL,free_index)
    if error < 0:
        print('Static Solver could not converge. (x perturbation)')

    # Get position of all nodes if Y is perturbed based on guess
    # Get position of all nodes if thetac is perturbed based on guess

    # Assemble jacobian matrix based on these perturbation values.

    # x_original = q_original[(nv+1)/2] # Get middle position

    # control_jacobian = [dxmid/dxc, dxmid/dyc, dxmid/dthetac;
    #                     dymid/dxc, dymid/dyc, dymid/dthetac]

    # J+ = Jt * (J * Jt)^-1

    # delta B = J+ * r  | where r is the residual

    x_arr = q_original[::2]  # Get X components for plotting
    y_arr = q_original[1::2]  # Get Y components for plotting

    h1 = plt.figure(1)
    plt.clf()  # Clear current figure
    plt.plot(x_arr, y_arr, 'ko-')
    plt.plot(x_mid, y_mid, 'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()

    xc_new = 0
    yc_new = 0
    thetac_new = 0

    return xc_new, yc_new, thetac_new, error