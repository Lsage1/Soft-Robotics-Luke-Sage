import numpy as np
from gradEb_hessEb import gradEb_hessEb
from PlotRodNetwork import PlotRodNetwork
import matplotlib.pyplot as plt
from to_woven import to_woven_j

def getFb_OO_tree(root_edge, root_vertex, vertexObjs, edgeObjs, EI, ndof_total):


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


def _tree_bending_recursive(edge_out, c_v, vertexObjs, edgeObjs, Fb, Jb, EI):
    """
    Recursive function to traverse tree and assemble bending forces and Jacobians.
    """

    edge_out.handled = True
    # Junction bending
    if c_v.junction:

        for edgePair, kappa_j in zip(c_v.edgePairs, c_v.junction_rest_kappa):
            edge0, edge1 = edgePair
            PlotRodNetwork(vertexObjs, edgeObjs, extra_vertices=None, extra_edges=[edge0, edge1])
            plt.show()
            # --- Tangents relative to junction ---
            def tangent_relative(edge, v):
                other = edge.get_other_vertex(v)
                t = np.array(other.coords) - np.array(v.coords)
                return t / np.linalg.norm(t)

            t0 = tangent_relative(edge0, c_v)
            t1 = tangent_relative(edge1, c_v)

            # --- Flip logic ---
            # Root >> branch: keep root orientation
            if edge0 is not edge_out:
                # Branch >> branch: flip one if both point outward
                if np.dot(t0, t1) > 0:
                    t0 = -t0  # virtual flip

            #  Virtual nodes for bending calculation
            dL = 0.5 * (edge0.rest_length + edge1.rest_length)
            node0 = np.array(c_v.coords) - dL * t0
            node1 = np.array(c_v.coords)
            node2 = np.array(c_v.coords) + dL * t1

            # Material directors
            m1e, m2e = edge0.m1, edge0.m2
            m1f, m2f = edge1.m1, edge1.m2

            # Compute force and Jacobian
            dF, dJ = gradEb_hessEb(node0, node1, node2, m1e, m2e, m1f, m2f, kappa_j, dL, EI)

            ind = np.concatenate([
                edge0.get_other_vertex(c_v).index,  # node0, 3 DOFs
                c_v.index,  # node1, 3 DOFs
                [edge0.theta_index],  # Edge0 twist, 1 DOF
                edge1.get_other_vertex(c_v).index,  # node2, 3 DOFs
                [edge1.theta_index]])  # Edge1 twist, 1 DOF
            Fb[ind] -= dF
            Jb[np.ix_(ind, ind)] -= dJ



    # Edge internal bending (ignore ends)
    elif not edge_out.network_root:
        edge_in = edge_out.parent
        v0 = edge_in.get_other_vertex(c_v).coords
        v1 = c_v.coords
        v2 = edge_out.get_other_vertex(c_v).coords
        #PlotRodNetwork(vertexObjs, edgeObjs, extra_vertices=None, extra_edges=[edge_in, edge_out])
        #plt.show()

        m1e, m2e = edge_in.m1, edge_in.m2
        m1f, m2f = edge_out.m1, edge_out.m2
        dL = 0.5 * (edge_in.rest_length + edge_out.rest_length)
        curvature0 = c_v.rest_kappa
        dF, dJ = gradEb_hessEb(v0, v1, v2, m1e, m2e, m1f, m2f, curvature0, dL, EI)

        ind = np.concatenate([
            edge_in.get_other_vertex(c_v).index,  # node0, 3 DOFs
            [edge_in.theta_index],  # Edge0 twist, 1 DOF
            c_v.index,  # node1, 3 DOFs
            [edge_out.theta_index],  # Edge1 twist, 1 DOF
            edge_out.get_other_vertex(c_v).index])  # node2, 3 DOFs

        Fb[ind] -= dF
        Jb[np.ix_(ind, ind)] -= dJ
        #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        #(to_woven_j(Jb, 3))
        #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    # Traverse children edges
    for edge_child in edge_out.children:
        if not edge_child.handled:
            v_next = edge_child.get_other_vertex(c_v)
            _tree_bending_recursive(edge_child, v_next, vertexObjs, edgeObjs, Fb, Jb, EI)






