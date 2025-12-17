def orient_edges_downhill(edge_in, v_in):
    """
    Recursively orient all edges so they point away from the root.

    After this pass:
      edge.vertex1 --> edge.vertex2
    always points downhill (away from root).
    """

    edge_in.handled = True
    # Ensure incoming edge points INTO v_in
    if edge_in.vertex2 == v_in:
        # Flip edge direction
        edge_in.vertex1, edge_in.vertex2 = edge_in.vertex2, edge_in.vertex1
    # Traverse outgoing edges
    for edge_out in v_in.edges:
        if edge_out.handled or edge_out is edge_in:
            continue

        v_next = edge_out.get_other_vertex(v_in)

        # Ensure edge_out points AWAY from v_in
        if edge_out.vertex1 is not v_in:
            edge_out.vertex1, edge_out.vertex2 = edge_out.vertex2, edge_out.vertex1

        # Recurse
        orient_edges_downhill(edge_out, v_next)
