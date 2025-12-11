import numpy as np
from gradEb_hessEb import gradEb_hessEb
from PlotRodNetwork import PlotRodNetwork

def getFb_OO_tree(root_edge, root_vertex, edgeObjs, vertexObjs, EI, ndof_total):
    print("computing bending")
    """
    Compute bending forces and Jacobian by recursively traversing the tree.

    Args:
        root_edge: EdgeObj at the start of the tree (root).
        root_vertex: VertexObj at the start of the tree.
        EI: bending stiffness
        ndof_total: total number of DOFs in the system

    Returns:
        Fb: global bending force vector (size = ndof_total)
        Jb: global bending Jacobian (size = ndof_total x ndof_total)
    """

    Fb = np.zeros(ndof_total)
    Jb = np.zeros((ndof_total, ndof_total))

    # Reset handled flags
    for e in edgeObjs:
        e.handled = False
    # Start recursion
    _tree_bending_recursive(root_edge, root_vertex, vertexObjs, edgeObjs, Fb, Jb, EI)

    # Sanity check
    assert Fb.shape[0] == ndof_total, "Fb vector length mismatch!"

    return Fb, Jb


def _tree_bending_recursive(edge_in, v_in, vertexObjs, edgeObjs, Fb, Jb, EI):
    """
    Recursive function to traverse tree and assemble bending forces and Jacobians.
    """
    edge_in.handled = True
    PlotRodNetwork(vertexObjs, edgeObjs, [v_in], [edge_in])

    # Junction bending
    if v_in.junction:
        for edgePair, kappa_j in zip(v_in.edgePairs, v_in.kappa_junction):
            edge0, edge1 = edgePair
            node0 = edge0.get_other_vertex(v_in).coords
            node1 = v_in.coords
            node2 = edge1.get_other_vertex(v_in).coords

            # Material directors
            m1e, m2e = edge0.m1, edge0.m2
            m1f, m2f = edge1.m1, edge1.m2

            # Voronoi length
            dL = 0.5 * (edge0.voronoi_length + edge1.voronoi_length)

            # Compute force and Jacobian
            dF, dJ = gradEb_hessEb(node0, node1, node2, m1e, m2e, m1f, m2f, kappa_j, dL, EI)

            # Assemble global indices
            ind = np.concatenate([edge0.vertex1.index, edge0.vertex2.index,
                                  v_in.index,
                                  edge1.vertex1.index, edge1.vertex2.index])
            Fb[ind] -= dF
            Jb[np.ix_(ind, ind)] -= dJ

    # Edge internal bending (ignore ends)
    if edge_in.is_internal():
        v0 = edge_in.prev_vertex(v_in).coords
        v1 = v_in.coords
        v2 = edge_in.next_vertex(v_in).coords

        m1e, m2e = edge_in.m1, edge_in.m2
        m1f, m2f = edge_in.m1_next, edge_in.m2_next
        dL = edge_in.voronoi_length
        curvature0 = edge_in.kappa

        dF, dJ = gradEb_hessEb(v0, v1, v2, m1e, m2e, m1f, m2f, curvature0, dL, EI)
        ind = np.concatenate([edge_in.vertex1.index, edge_in.vertex2.index, v_in.index])
        Fb[ind] -= dF
        Jb[np.ix_(ind, ind)] -= dJ

    # Traverse children edges
    for edge_out in edge_in.children:
        if not edge_out.handled:
            v_next = edge_out.get_other_vertex(v_in)
            _tree_bending_recursive(edge_out, v_next, vertexObjs, edgeObjs, Fb, Jb, EI)






