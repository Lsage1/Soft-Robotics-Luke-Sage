import numpy as np

def to_woven(q, n):
    """
    [x1,y1,z1, x2,y2,z2, ..., xn,yn,zn, θ1..θ(n-1)]
    →
    [x1,y1,z1,θ1,
     x2,y2,z2,θ2,
     ...
     x(n-1),y(n-1),z(n-1),θ(n-1),
     xn,yn,zn]
    """

    q = np.asarray(q)

    positions = q[:3*n]
    thetas = q[3*n:]  # length n-1

    q_out = []

    for i in range(n):
        # position
        q_out.extend(positions[3*i:3*i+3])

        # twist only for first n-1 nodes
        if i < n - 1:
            q_out.append(thetas[i])

    return np.array(q_out)

def to_woven_j(J_der, n):
    """
    General DER → interleaved Jacobian reordering
    """

    idx = []

    # positions + theta per node
    for i in range(n):
        # positions
        idx.extend([3*i, 3*i + 1, 3*i + 2])

        # theta for first n-1 nodes
        if i < n - 1:
            idx.append(3*n + i)

    return J_der[np.ix_(idx, idx)]
