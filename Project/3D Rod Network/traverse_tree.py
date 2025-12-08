def traverse_tree(start_edge, start_vertex, edgeObjs, callback=None):
    """
    Generic tree traversal for your rod network.

    Calls `callback(parent_edge, current_edge, vertex_entered)`
    in the exact order edges are discovered.

    Parameters
    ----------
    start_edge : Edge
        Edge where traversal begins.
    start_vertex : Vertex
        The vertex on the start edge we treat as incoming direction.
    edgeObjs : list[Edge]
        All edges in the network.
    callback : function(parent, edge, vertex)
        Called every time we step onto a new edge.
        parent = previous edge (None on the first step)
        edge   = newly discovered edge
        vertex = vertex where we entered this edge
    """

    # Reset flags
    for edge in edgeObjs:
        edge.handled = False

    activeJunction = []
    tree_search_active = True

    # The currently active edge and vertex we are coming from
    edge01 = start_edge
    vertex0 = start_vertex

    # The first edge has no parent
    parent_edge = None

    print("Starting Tree Pass")

    # Mark start edge handled
    start_edge.handled = True

    # Call callback for start edge
    if callback is not None:
        callback(None, start_edge, start_vertex)

    while tree_search_active:

        # Move across edge01 to reach vertex1
        vertex1 = edge01.get_other_vertex(vertex0)

        # If this is a junction, we will need to remember it
        if vertex1.junction:
            remaining = [e for e in vertex1.edges
                         if (not e.handled and e is not edge01)]
            if len(remaining) > 0 and all(j[0] != vertex1 for j in activeJunction):
                activeJunction.append([vertex1, edge01])

        # Try to find next unhandled edge
        edge12, found_edge = vertex1.get_unhandled_edge(edge01)

        if not found_edge:
            # Backtrack
            if len(activeJunction) > 0:
                while len(activeJunction) > 0:
                    vertex1, edge01 = activeJunction[-1]
                    edge12, found_edge = vertex1.get_unhandled_edge(edge01)
                    if found_edge:
                        vertex0 = edge01.get_other_vertex(vertex1)
                        break
                    else:
                        activeJunction.pop()
                else:
                    break   # No more work
            else:
                break

        # We found a new edge (edge12). Mark parent and call callback.
        edge12.root = edge01  # parent

        vertex2 = edge12.get_other_vertex(vertex1)

        edge12.handled = True

        if callback is not None:
            callback(edge01, edge12, vertex1)

        # Step to next edge
        parent_edge = edge01
        edge01 = edge12
        vertex0 = vertex1
        vertex1 = vertex2