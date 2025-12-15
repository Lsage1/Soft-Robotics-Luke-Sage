def assign_dof_indices(vertexObjs, edgeObjs):
    dof = 0
    ne = len(edgeObjs)
    nv = len(vertexObjs)

    # There must be exactly nv - 1 edges if the alternating pattern is required
    if ne != nv - 1:
        raise ValueError(
            f"Expected {nv-1} edges for {nv} vertices (vertex-edge-vertex pattern), "
            f"but got {ne}."
        )

    # Vertex-edge-vertex pattern
    for i, v in enumerate(vertexObjs):
        # Assign 3 DOFs to vertex
        v.test_index = [dof, dof+1, dof+2]
        dof += 3

        # If not the last vertex, assign an edge DOF after it
        if i < ne:
            e = edgeObjs[i]
            e.test_index = dof
            dof += 1



    return dof
