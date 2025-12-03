import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def PlotRodNetwork(vertices, edges):

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    # Plot vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=40)

    # Plot edges
    for e in edges:
        p1 = vertices[e[0]]
        p2 = vertices[e[1]]
        xs = [p1[0], p2[0]]
        ys = [p1[1], p2[1]]
        zs = [p1[2], p2[2]]
        ax.plot(xs, ys, zs)

    # Make axes equal scale
    max_range = np.array([vertices[:,0].max()-vertices[:,0].min(),
                          vertices[:,1].max()-vertices[:,1].min(),
                          vertices[:,2].max()-vertices[:,2].min()]).max() / 2.0

    mid = vertices.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()