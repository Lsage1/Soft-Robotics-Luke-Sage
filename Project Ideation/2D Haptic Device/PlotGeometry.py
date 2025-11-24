import numpy as np
import matplotlib.pyplot as plt


def PlotGeometry(vertices, edgeIndex, junctions=None):
    """
    Plot vertices and edges of a discrete rod network.

    Parameters:
        vertices (np.ndarray): Array of vertex coordinates, shape (N, 2)
        edgeIndex (np.ndarray): Array of edges, shape (M, 2)
        junctions (list or np.ndarray, optional): Indices of junction vertices to highlight
    """
    plt.figure(figsize=(6, 6))

    # Plot edges
    for edge in edgeIndex:
        v0, v1 = vertices[edge[0]], vertices[edge[1]]
        plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'b-', lw=2)

    # Plot vertices
    plt.scatter(vertices[:, 0], vertices[:, 1], color='red', s=50, zorder=5)

    # Highlight junctions if provided
    if junctions is not None:
        plt.scatter(vertices[junctions, 0], vertices[junctions, 1], color='green', s=100, label='Junctions', zorder=6)

    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Discrete Rod Network')
    if junctions is not None:
        plt.legend()
    plt.show()


