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
        for i, (edgePair, kappa_j) in enumerate(zip(c_v.edgePairs, c_v.junction_rest_kappa)):
            edge0, edge1 = edgePair
            print(edge0.id, edge1.id)

            #PlotRodNetwork(vertexObjs, edgeObjs, extra_vertices=[c_v], extra_edges=[edge0, edge1])
            #plt.title("Junction")
            #plt.show()

            # Get tangents relative to junction
            def tangent_relative(edge, v):
                other = edge.get_other_vertex(v)
                t = np.array(other.coords) - np.array(v.coords)
                return t / np.linalg.norm(t)

            t0_away = tangent_relative(edge0, c_v)
            t1_away = tangent_relative(edge1, c_v)

            # --- Replay rest-time flip ---
            flip0 = c_v.junction_flip0[i]
            flip1 = c_v.junction_flip1[i]# use stored decision from rest

            t0_bending = -t0_away if flip0 else t0_away
            t1_bending = -t1_away if flip1 else t1_away

            #  Virtual nodes for bending calculation
            dL = 0.5 * (edge0.rest_length + edge1.rest_length)
            node0 = np.array(c_v.coords) - dL * t0_bending
            node1 = np.array(c_v.coords)
            node2 = np.array(c_v.coords) + dL * t1_bending

            edge0_points_to_junction = (edge0.vertex2 == c_v)

            # Material directors
            m1e, m2e = edge0.m1.copy(), edge0.m2.copy()
            m1f, m2f = edge1.m1.copy(), edge1.m2.copy()

            # For edge0 material directors:
            # Compute the tangent that gradEb_hessEb will use
            te_actual = (node1 - node0) / np.linalg.norm(node1 - node0)

            # Check if edge0.tangent aligns with te_actual
            if np.dot(edge0.tangent, te_actual) < 0:
                # They point in opposite directions, so flip material directors
                m1e = -m1e
                m2e = -m2e

                # For edge1 material directors:
            tf_actual = (node2 - node1) / np.linalg.norm(node2 - node1)

            if np.dot(edge1.tangent, tf_actual) < 0:
                # They point in opposite directions, so flip material directors
                m1f = -m1f
                m2f = -m2f

            # In junction bending, add this before gradEb_hessEb call:
            print(f"\n=== Junction {c_v.id}, Pair {i}: Edges ({edge0.id}, {edge1.id}) ===")
            print(f"node0: {node0}, node1: {node1}, node2: {node2}")
            print(f"te_actual: {te_actual}, edge0.tangent: {edge0.tangent}")
            print(f"tf_actual: {tf_actual}, edge1.tangent: {edge1.tangent}")
            print(f"te·edge0.tangent = {np.dot(edge0.tangent, te_actual):.4f}")
            print(f"tf·edge1.tangent = {np.dot(edge1.tangent, tf_actual):.4f}")
            print(f"m1e before: {edge0.m1}, after: {m1e}")
            print(f"m1f before: {edge1.m1}, after: {m1f}")
            print(f"kappa_j (rest): {kappa_j}")


            # Compute force and Jacobian
            dF, dJ = gradEb_hessEb(node0, node1, node2, m1e, m2e, m1f, m2f, kappa_j, dL, EI)

            ind = np.concatenate([
                edge0.get_other_vertex(c_v).index,  # node0, 3 DOFs
                [edge0.theta_index],                # Edge0 twist, 1 DOF
                c_v.index,                          # node1, 3 DOFs
                [edge1.theta_index],                # Edge1 twist, 1 DOF
                edge1.get_other_vertex(c_v).index   # node2, 3 DOFs
            ])

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






