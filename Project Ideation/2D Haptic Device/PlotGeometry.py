import numpy as np
import matplotlib.pyplot as plt

def PlotGeometry(q0, edgeIndex):
    """
    Plot vertices and edges of a discrete rod network.

    Parameters:
        q0 (np.ndarray): Linearized vertex coordinates, shape (2*N,)
        edgeIndex (np.ndarray): Array of edges, shape (M, 2)
    """
    # Reshape q0 to (N, 2) for plotting
    vertices = q0.reshape(-1, 2)



    # Plot edges
    for edge in edgeIndex:
        v0, v1 = vertices[edge[0]], vertices[edge[1]]
        plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'b-', lw=2)

    # Plot vertices
    plt.scatter(vertices[:, 0], vertices[:, 1], color='red', s=50, zorder=5)

    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Discrete Rod Network')