import numpy as np

def ComputeTangentEdges(edges, curr):

    # edges: list of EdgeObj instances
    # Computes the tangent for each edge and stores it in edge.tangent
    # If Curr == True, use current coordinates, return to edge.tangent
    # If Curr == False, use previous coordinates, return to edge.tangent0

    for edge in edges:
        if curr:
            p0 = np.array(edge.vertex1.coords)
            p1 = np.array(edge.vertex2.coords)
        else:
            p0 = np.array(edge.vertex1.coords0)
            p1 = np.array(edge.vertex2.coords0)


        e = p1 - p0
        n = np.linalg.norm(e)

        if curr:
            edge.tangent = e / n
        else:
            edge.tangent0 = e / n

    return edges