import numpy as np

def to_interleaved(q, n):
    """
    Convert:
    [x1,y1,z1,θ1, x2,y2,z2,θ2, ..., xn,yn,zn,θn]
    →
    [x1,y1,z1, x2,y2,z2, ..., xn,yn,zn, θ1..θ(n-1)]
    """

    q = np.asarray(q)

    positions = []
    thetas = []

    for i in range(n):
        base = 4 * i
        positions.extend(q[base:base + 3])   # xi, yi, zi
        if i < n - 1:
            thetas.append(q[base + 3])        # θi

    return np.array(positions + thetas)
