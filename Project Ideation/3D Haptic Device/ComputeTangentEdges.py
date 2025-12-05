import numpy as np

def ComputeTangentEdges(edges):

    # edges: list of EdgeObj instances
    # Computes the tangent for each edge and stores it in edge.tangent

    for edge in edges:
        p0 = np.array(edge.vertex1.coords)
        p1 = np.array(edge.vertex2.coords)

        e = p1 - p0
        n = np.linalg.norm(e)


        edge.tangent = e / n

    return edges