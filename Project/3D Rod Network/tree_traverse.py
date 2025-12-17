from computeSpaceParallel_OO import computeSpaceParallel_OO
from computeMaterialDirectors_OO import computeMaterialDirectors_OO
from PlotRodNetwork import PlotRodNetwork
from getKappa_OO import getKappa_OO
import numpy as np
from computeKappa import computeKappa

def tree_CSP_CMD(edge_in, vertex_from, vertexObjs, edgeObjs):
    """
    Recursive equivalent of your while-loop + activeJunction stack.
    edge_in: the edge we arrived on
    vertex_from: the vertex we came from
    """

    vertex_curr = edge_in.get_other_vertex(vertex_from)
    edge_in.handled = True


    # Get all unhandled outgoing edges except the one we came from
    for edge_out in vertex_curr.edges:

        if edge_out.handled:
            continue
        if edge_out is edge_in:
            continue


        # 1. Do your per-edge work here:
        computeSpaceParallel_OO(edge_in, edge_out)
        computeMaterialDirectors_OO(edge_out)
        edge_out.handled = True

        # 2. Recurse outward
        tree_CSP_CMD(edge_out, vertex_curr, vertexObjs, edgeObjs)



########################################################################################################################

def tree_getKappa(edge_in, v_in, vertexObjs, edgeObjs):


    """
    Recursive curvature pass with junction handling.

    edge_in : the edge we arrived on (root/incoming)
    v_in    : the vertex we arrived at
    """
    edge_in.handled = True

    # --- Handle junction curvature ---
    if v_in.junction:
        v_in.junction_kappa = []

        # Loop over all pairs of edges at junction
        for edge0, edge1 in v_in.edgePairs:
            v0 = edge0.get_other_vertex(v_in)
            v2 = edge1.get_other_vertex(v_in)

            t0 = edge0.tangent
            t1 = edge1.tangent

            # Extract m1 and m2 for the current and previous edges
            m1e = edge0.m1  # m1 vector on previous  edge
            m2e = edge0.m2  # m2 vector on previous edge
            m1f = edge1.m1  # m1 vector on current edge
            m2f = edge1.m2  # m2 vector on current edge

            node0 = np.array(v0.coords)
            node1 = np.array(v_in.coords)
            node2 = np.array(v2.coords)

            # --- Compute rest curvature using existing helper ---
            kappa_local = computeKappa(node0, node1, node2, m1e, m2e, m1f, m2f)
            v_in.junction_kappa.append(kappa_local)

    # --- Normal recursive pass for child edges ---
    for edge_out in v_in.edges:
        if edge_out.handled or edge_out is edge_in:
            continue

        v_next = edge_out.get_other_vertex(v_in)
        v_prev = edge_in.get_other_vertex(v_in)

        # Compute curvature along the current edge
        v_in.kappa = getKappa_OO(np.array(v_prev.coords), np.array(v_in.coords), np.array(v_next.coords), edge_in, edge_out)

        edge_out.handled = True

        # Recurse
        tree_getKappa(edge_out, v_next, vertexObjs, edgeObjs)




