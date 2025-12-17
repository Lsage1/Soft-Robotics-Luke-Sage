import numpy as np
from gradEb_hessEb import gradEb_hessEb
from PlotRodNetwork import PlotRodNetwork
import matplotlib.pyplot as plt
from to_woven import to_woven_j

def getFb_OO_tree(root_vertex, vertexObjs, edgeObjs, EI, ndof_total):


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

    for v in vertexObjs:
        v.handled = False

    root_edge = None

    # Start recursion
    _tree_bending_recursive(root_vertex, root_edge, vertexObjs, edgeObjs, Fb, Jb, EI)

    # Sanity check
    assert Fb.shape[0] == ndof_total, "Fb vector length mismatch!"

    return Fb, Jb

 #####################################################################################################

 #####################################################################################################

def _tree_bending_recursive(c_v, root_edge, vertexObjs, edgeObjs, Fb, Jb, EI):
    """
    Recursive function to traverse tree and assemble bending forces and Jacobians.
    """


    # Junction bending
    if c_v.junction:
        #print("got a junction")
        #print(c_v.junction_rest_kappa, c_v.edgePairs)
        for edgePair, kappa_j in zip(c_v.edgePairs, c_v.junction_rest_kappa):
            edge0, edge1 = edgePair

            # Get the "other" vertices for each edge
            v0 = edge0.get_other_vertex(c_v)
            v2 = edge1.get_other_vertex(c_v)
            #PlotRodNetwork(vertexObjs, edgeObjs, extra_vertices=[c_v], extra_edges=[edge0, edge1])
            #plt.title("Junction")
            #plt.show()

            t0 = edge0.tangent
            t1 = edge1.tangent

            if not edge0 == root_edge and not edge1 == root_edge:
                t0 = -t0

            # Extract m1 and m2 for the current and previous edges
            m1e = edge0.m1  # m1 vector on previous  edge
            m2e = edge0.m2  # m2 vector on previous edge
            m1f = edge1.m1  # m1 vector on current edge
            m2f = edge1.m2  # m2 vector on current edge
            dL = 0.5 * (edge0.rest_length + edge1.rest_length)

            node0 = np.array(v0.coords)
            node1 = np.array(c_v.coords)
            node2 = np.array(v2.coords)
            #print(node0, node1, node2)


            dF, dJ = gradEb_hessEb(node0, node1, node2, m1e, m2e, m1f, m2f, kappa_j, dL, EI)
            #print(dJ)

            ind = np.concatenate([
                edge0.get_other_vertex(c_v).index,  # node0, 3 DOFs
                [edge0.theta_index],                # Edge0 twist, 1 DOF
                c_v.index,                          # node1, 3 DOFs
                [edge1.theta_index],                # Edge1 twist, 1 DOF
                edge1.get_other_vertex(c_v).index])  # node2, 3 DOFs

            #print(ind)
            Fb[ind] -= dF
            Jb[np.ix_(ind, ind)] -= dJ

    # Edge internal bending (ignore ends)
    elif root_edge != None and not c_v.end:
        #print("got a bending joint")

        edge_out = c_v.get_other_edge(root_edge)
        edge_in = root_edge

        #PlotRodNetwork(vertexObjs, edgeObjs, extra_vertices=[c_v], extra_edges=[edge_in, edge_out])
        #plt.title("regular_bending")
        plt.show()

        v0 = edge_in.get_other_vertex(c_v).coords
        v1 = c_v.coords
        v2 = edge_out.get_other_vertex(c_v).coords
        #print(edge_out.vertex1.coords, edge_out.vertex2.coords, " --> ", edge_in.vertex1.coords, edge_in.vertex2.coords)



        m1e, m2e = edge_in.m1, edge_in.m2
        m1f, m2f = edge_out.m1, edge_out.m2
        dL = 0.5 * (edge_in.rest_length + edge_out.rest_length)
        curvature0 = c_v.rest_kappa

        dF, dJ = gradEb_hessEb(v0, v1, v2, m1e, m2e, m1f, m2f, curvature0, dL, EI)
        #(dJ)
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



    c_v.handled = True
    for edge_child in c_v.edges:

        next_vertex = edge_child.get_other_vertex(c_v)
        if not next_vertex.handled:
            # print("V_Next",  v_next.coords)

            _tree_bending_recursive(next_vertex, edge_child, vertexObjs, edgeObjs, Fb, Jb, EI)






