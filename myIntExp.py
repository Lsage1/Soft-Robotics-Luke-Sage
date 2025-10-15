import numpy as np
from getForceJacobian import getForceJacobian


# An explicit integrator to replace implicit integrator myInt.py

def myIntExp(t_new, x_old, u_old, free_DOF, stiffness_matrix, index_matrix, m, dt, l_k):


    x_new = x_old + u_old * dt
    print("->", x_new, u_old)

    f, J = getForceJacobian(x_new, x_old, u_old, stiffness_matrix, index_matrix, m, dt, l_k)
    #print("f = ", f)
    f_free = f[free_DOF]
    #print("f_free = ", f_free)

    u_new = u_old
    x_new = x_old
    print(m)
    u_new[free_DOF] = u_old[free_DOF] - (f[free_DOF] / m[free_DOF]) * dt
    x_new[free_DOF] = x_old[free_DOF] + u_new[free_DOF] * dt

    return x_new, u_new