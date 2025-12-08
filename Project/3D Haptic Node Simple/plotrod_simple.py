import numpy as np
import matplotlib.pyplot as plt
from set_axes_equal import set_axes_equal

def plotrod_simple(q1, q2, q3, ctime):
    """
    Plot 3 rods (given as q vectors) in a single 3D figure.

    Parameters:
    - q1, q2, q3 : Position DOF vectors for rods
    - ctime     : Current time (for plot title)
    """

    def extract_xyz(q):
        """Extract x,y,z coordinates from a q vector."""
        x = q[0::4]
        y = q[1::4]
        z = q[2::4]
        return x, y, z

    # Extract node coordinates
    x1, y1, z1 = extract_xyz(q1)
    x2, y2, z2 = extract_xyz(q2)
    x3, y3, z3 = extract_xyz(q3)

    # Plot
    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')

    # Rod 1 (black)
    ax.plot3D(x1, y1, z1, 'k-o')
    ax.plot3D([x1[0]], [y1[0]], [z1[0]], 'r^')  # first node
    ax.plot3D([x1[-1]], [y1[-1]], [z1[-1]], 'b^')  # last node

    # Rod 2 (blue)
    ax.plot3D(x2, y2, z2, 'b-o')
    ax.plot3D([x2[0]], [y2[0]], [z2[0]], 'r^')
    ax.plot3D([x2[-1]], [y2[-1]], [z2[-1]], 'b^')

    # Rod 3 (green)
    ax.plot3D(x3, y3, z3, 'g-o')
    ax.plot3D([x3[0]], [y3[0]], [z3[0]], 'r^')
    ax.plot3D([x3[-1]], [y3[-1]], [z3[-1]], 'b^')

    # Titles and labels
    ax.set_title(f"t = {ctime:.2f}")
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    set_axes_equal(ax)
    plt.draw()
