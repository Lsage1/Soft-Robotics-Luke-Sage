import numpy as np
from computeKappa import computeKappa

def getKappa_OO(vertex, end_edge, edgeObjs, vertexObjs):
    print(vertex.coords)
    vertex0 = vertex
    edge01 = end_edge

    vertex1 = edge01.get_other_vertex(vertex0)
    edge12 = vertex1.get_other_edge(edge01) ### EDGE 12 is considered the "Current" edge

    vertex2 = edge12.get_other_vertex(vertex1)

    # Make sure there are enough edges per discrete rod
    if vertex1.end or vertex1.junction:
        raise RuntimeError("Discrete rod has 2 or fewer edges, cannot compute Kappa.")

    active = True
    while active:  # Loop over edges, starting from an end point until a junction

        if vertex1.end:
            break # if the middle vertex is an end, we can exit

        if vertex1.junction_handled:
            break # if another process has already calculated gammas for the junction, we can exit.

        elif vertex1.junction: # If vertex2 is a junction, we have a messier problem to deal with.
            # A solution is to get a kappa between every other edge along the junction, the average them out.

            for edgePair in vertex.edgePairs:  # Get all the attached edges one by one

                if edgePair[0] == edge01: # First make sure we get the other edge in each edge pair
                    edge12 = edgePair[1]
                else:
                    edge12 = edgePair[0]
                # then, calculate kappa for this edge pair, and add it to the second edge's kappa value
                vertex2 = edge12.get_other_vertex(vertex1)

                # Get the kappa with respect to the other edge in the pair
                kappa_local = computeKappa(np.array(vertex0.coords), np.array(vertex1.coords), np.array(vertex2.coords), edge01.m1, edge01.m2, edge12.m1, edge12.m2)
                # Edges adjacent to a junction will have multiple
                edge12.kappa.append(kappa_local)

            vertex1.junction_handled = True # Mark this junction as handled, it will be ignored next time

        else: # Since junctions and ends have been considered, there will be only one other edge
            edge12 = vertex1.get_other_edge(edge01)
            vertex2 = edge12.get_other_vertex(vertex1)
            # Compute local curvature at each node
            kappa_local = computeKappa(np.array(vertex0.coords), np.array(vertex1.coords), np.array(vertex2.coords), edge01.m1, edge01.m2, edge12.m1, edge12.m2)
            edge12.kappa = kappa_local

        # Step to next pair of edges.
        vertex0 = vertex1
        vertex1 = vertex2
        edge01 = edge12
        # New edge12 and new vertex2 will be handled in the next loop

    return