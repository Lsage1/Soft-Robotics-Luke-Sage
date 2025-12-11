import numpy as np
from computeReferenceTwist import computeReferenceTwist
from PlotRodNetwork import PlotRodNetwork

def getRefTwist_OO(vertexObjs, edgeObjs, end_edge, first_end_vertex):
    # Reset handled flags
    for e in edgeObjs:
        e.handled = False



    # Start recursion
    tree_getRefTwist(end_edge, first_end_vertex, vertexObjs, edgeObjs)

def tree_getRefTwist(edge_in, v_in, vertexObjs, edgeObjs):
    """
    Recursive computation of reference twist.
    edge_in : edge we arrived on
    v_in    : vertex we arrived at
    """
    edge_in.handled = True

    # For each child edge
    for edge_out in edge_in.children:

        if edge_out.handled:
            continue

        v_curr = v_in
        v_next = edge_out.get_other_vertex(v_curr)

        # --- Compute reference twist for this branch ---
        a1e = edge_in.a1
        a1f = edge_out.a1
        t1  = edge_in.tangent
        t2  = edge_out.tangent

        branch_idx = edge_out.get_branch_number()

        # Ensure reference twist vector is initialized
        if v_curr.ref_twist is None or len(v_curr.ref_twist) < len(edge_in.children):
            v_curr.ref_twist = np.zeros(len(edge_in.children))

        v_curr.ref_twist[branch_idx] = computeReferenceTwist(
            a1e, a1f, t1, t2, v_curr.ref_twist[branch_idx]
        )

        # Optional: for debugging / visualization
        # PlotRodNetwork(vertexObjs, edgeObjs, [], [edge_in, edge_out])

        # Mark and recurse
        edge_out.root = edge_in
        edge_out.handled = True
        tree_getRefTwist(edge_out, v_next, vertexObjs, edgeObjs)

