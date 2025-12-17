from computeSpaceParallel_OO import computeSpaceParallel_OO
from computeMaterialDirectors_OO import computeMaterialDirectors_OO
from PlotRodNetwork import PlotRodNetwork
from getKappa_OO import getKappa_OO
import numpy as np

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

        edges_at_junction = v_in.edges

        # Tangents relative to junction
        def tangent_relative(edge, v):
            other = edge.get_other_vertex(v)
            t = np.array(other.coords) - np.array(v.coords)
            return t / np.linalg.norm(t)

        # Loop over all pairs of edges at junction
        for i, (edge0, edge1) in enumerate(v_in.edgePairs):
            flip0 = v_in.junction_flip0[i]
            flip1 = v_in.junction_flip1[i]


            t0 = tangent_relative(edge0, v_in)
            t1 = tangent_relative(edge1, v_in)

            if flip0:
                t0 = -t0
            if flip1:
                t1 = -t1

            # --- Construct virtual nodes for curvature ---
            L = 0.5 * (edge0.rest_length + edge1.rest_length)  # or Voronoi length
            node0 = np.array(v_in.coords) - L * t0
            node1 = np.array(v_in.coords)
            node2 = np.array(v_in.coords) + L * t1

            # --- Compute rest curvature using existing helper ---
            kappaBar = getKappa_OO(node0, node1, node2, edge0, edge1)
            v_in.junction_kappa.append(kappaBar)

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




