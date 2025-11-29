import numpy as np

def DetectJunctions(vertices, edgeIndex):

    # Count the number of edges connected to each vertex
    num_vertices = vertices.shape[0]
    edge_counts = np.zeros(num_vertices, dtype=int)

    for edge in edgeIndex:
        edge_counts[edge[0]] += 1
        edge_counts[edge[1]] += 1

    # Junctions are vertices connected to more than 2 edges
    junctions = np.where(edge_counts > 2)[0]
    return junctions