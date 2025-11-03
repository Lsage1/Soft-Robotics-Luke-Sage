from staticSolver import staticSolver
import matplotlib.pyplot as plt
import numpy as np

def getDirichlet(xc, yc, thetac, x_mid, y_mid, nv,
                 q0, # guess beam positions from last time step.
                 tol, maximum_iter, EI, EA, W, deltaL, free_index):

    error = tol * 10  # error
    q_guess = q0.copy

    while error > tol:
        dx = 10e-2 * deltaL # X perturbation value
        dy = 10e-2 * deltaL # y perturbation value
        dtheta = 10e-3 # Theta perturbation value

        # Control Jacobian:
        J_c = np.zeros((2, 3))


        # Get reference position of all nodes given guess end conditions xc, yc, thetac.
        # Set the boundary conditions to a guess from the last frames.
        q_guess[nv * 2 - 2] = xc  # x_n
        q_guess[nv * 2 - 1] = yc  # y_n
        q_guess[nv * 2 - 4] = xc - deltaL * np.cos(thetac)  # X n-1
        q_guess[nv * 2 - 3] = yc - deltaL * np.sin(thetac)  # Y n-1
        print(x_mid, y_mid)
        q_original, err = staticSolver(q_guess, tol, maximum_iter, EI, EA, W, deltaL,free_index)
        if err < 0:
            print('Static Solver could not converge.')

        #print("beam middle position: ", q_original[nv-1], q_original[nv])
        residual = np.array([[x_mid - q_original[nv-1]], [y_mid - q_original[nv]]] )
        print("residual: ")
        print(residual)

        # Get position of all nodes if X is perturbed based on guess
        q0_xpert = q_guess.copy()
        q0_xpert[nv * 2 - 2] = xc + dx  # x_n
        q0_xpert[nv * 2 - 1] = yc  # y_n
        q0_xpert[nv * 2 - 4] = xc + dx - deltaL * np.cos(thetac)  # X n-1
        q0_xpert[nv * 2 - 3] = yc - deltaL * np.sin(thetac)  # Y n-1
        q_p_x, err = staticSolver(q0_xpert, tol, maximum_iter, EI, EA, W, deltaL,free_index)
        if err < 0:
            print('Static Solver could not converge. (x perturbation)')

        q0_ypert = q_guess.copy()
        q0_ypert[nv * 2 - 2] = xc  # x_n
        q0_ypert[nv * 2 - 1] = yc + dy  # y_n
        q0_ypert[nv * 2 - 4] = xc - deltaL * np.cos(thetac)  # X n-1
        q0_ypert[nv * 2 - 3] = yc + dy - deltaL * np.sin(thetac)  # Y n-1
        q_p_y, err = staticSolver(q0_ypert, tol, maximum_iter, EI, EA, W, deltaL, free_index)
        if err < 0:
            print('Static Solver could not converge. (y perturbation)')

        q0_thetapert = q_guess.copy()
        q0_thetapert[nv * 2 - 2] = xc  # x_n
        q0_thetapert[nv * 2 - 1] = yc  # y_n
        q0_thetapert[nv * 2 - 4] = xc - deltaL * np.cos(thetac + dtheta)  # X n-1
        q0_thetapert[nv * 2 - 3] = yc - deltaL * np.sin(thetac + dtheta)  # Y n-1
        q_p_theta, error = staticSolver(q0_thetapert, tol, maximum_iter, EI, EA, W, deltaL, free_index)
        if error < 0:
            print('Static Solver could not converge. (theta perturbation)')

        print(q_original[nv-1], q_p_x[nv-1])

        # Assemble Jacobian Matrix
        J_c[0,0] = (q_p_x[nv-1] - q_original[nv-1]) / dx
        J_c[1,0] = (q_p_x[nv] - q_original[nv]) / dx

        J_c[0,1] = (q_p_y[nv-1] - q_original[nv-1]) / dy
        J_c[1,1] = (q_p_y[nv] - q_original[nv]) / dy

        J_c[0,2] = (q_p_theta[nv-1] - q_original[nv-1]) / dtheta
        J_c[1,2] = (q_p_theta[nv] - q_original[nv]) / dtheta

        print(J_c)

        j_pinv = np.linalg.pinv(J_c)
        b = j_pinv @ residual

        print(b)


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