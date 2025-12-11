import numpy as np
from ComputeTangentEdges import ComputeTangentEdges
from parallel_transport import parallel_transport



def computeTimeParallel_OO(end_edge, first_end_vertex, edgeObjs, q0, q):
    """
    Single recursive function that performs the entire time-parallel frame update.
    Traversal is driven entirely by edge.children.
    """

    # ------------------------------------------------------------
    # Precompute tangents for all edges (old and new shapes)
    # ------------------------------------------------------------
    ComputeTangentEdges(edgeObjs, curr=True)   # sets edge.tangent
    ComputeTangentEdges(edgeObjs, curr=False)    # sets edge.tangent0

    # Reset handled flags
    for e in edgeObjs:
        e.handled = False

    # ------------------------------------------------------------
    # Compute the initial edge (root) frame
    # ------------------------------------------------------------
    t0 = end_edge.tangent0
    t1 = end_edge.tangent

    end_edge.a1 = parallel_transport(end_edge.a1_old, t0, t1)
    end_edge.a1 = end_edge.a1 - np.dot(end_edge.a1, t1) * t1
    end_edge.a1 /= np.linalg.norm(end_edge.a1)

    end_edge.a2 = np.cross(t1, end_edge.a1)
    end_edge.a2 = end_edge.a2 - np.dot(end_edge.a2, t1) * t1
    end_edge.a2 /= np.linalg.norm(end_edge.a2)

    end_edge.handled = True

    # ------------------------------------------------------------
    # Launch recursion
    # ------------------------------------------------------------
    tree_timeparallel_recursive(end_edge, first_end_vertex)


def tree_timeparallel_recursive(edge_in, v_in):
    """
    Recursive time-parallel pass.
    This is the ONLY recursive tree function needed.
    """

    # For every outgoing branch (children edges)
    for edge_out in edge_in.children:

        if edge_out.handled:
            continue

        # ----- Compute time-parallel transported frame on edge_out -----

        t0 = edge_out.tangent0
        t1 = edge_out.tangent

        edge_out.a1 = parallel_transport(edge_out.a1_old, t0, t1)
        edge_out.a1 = edge_out.a1 - np.dot(edge_out.a1, t1) * t1
        edge_out.a1 /= np.linalg.norm(edge_out.a1)

        edge_out.a2 = np.cross(t1, edge_out.a1)
        edge_out.a2 = edge_out.a2 - np.dot(edge_out.a2, t1) * t1
        edge_out.a2 /= np.linalg.norm(edge_out.a2)

        edge_out.handled = True
        edge_out.root = edge_in

        # Continue traversal
        v_next = edge_out.get_other_vertex(v_in)
        tree_timeparallel_recursive(edge_out, v_next)
