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

    # ---- Leaving the function performs backtracking ----

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
        print("junction going")
        v_in.junction_kappa = []

        edges_at_junction = v_in.edges

        # Loop over all pairs of edges at junction
        for i, edgeA in enumerate(edges_at_junction):
            for j, edgeB in enumerate(edges_at_junction):
                if j <= i:
                    continue  # skip duplicates/self-pair

                # --- Determine tangent vectors relative to junction ---
                def tangent_relative_to_junction(edge, v):
                    other = edge.get_other_vertex(v)
                    t = np.array(other.coords) - np.array(v.coords)
                    return t / np.linalg.norm(t)

                tA = tangent_relative_to_junction(edgeA, v_in)
                tB = tangent_relative_to_junction(edgeB, v_in)

                # --- Decide if flip is needed ---
                # Root → branch: no flip
                if edgeA is not edge_in:
                    # Branch → branch: flip one if both point outward
                    if np.dot(tA, tB) > 0:
                        tA = -tA  # virtual flip

                # --- Construct virtual nodes for curvature ---
                L = 0.5 * (edgeA.rest_length + edgeB.rest_length)  # or Voronoi length
                node0 = np.array(v_in.coords) - L * tA
                node1 = np.array(v_in.coords)
                node2 = np.array(v_in.coords) + L * tB

                # --- Compute rest curvature using existing helper ---
                kappaBar = getKappa_OO(node0, node1, node2, edgeA, edgeB)
                v_in.junction_kappa.append(kappaBar)

    # --- Normal recursive pass for child edges ---
    for edge_out in v_in.edges:
        if edge_out.handled or edge_out is edge_in:
            continue

        v_next = edge_out.get_other_vertex(v_in)
        v_prev = edge_in.get_other_vertex(v_in)

        # Compute curvature along the current edge
        v_in.kappa = getKappa_OO(np.array(v_prev.coords), np.array(v_in.coords), np.array(v_next.coords), edge_in, edge_out)

        # Track tree structure
        edge_in.children.append(edge_out)
        edge_out.parent = edge_in

        # Allocate ref_twist if branching
        unhandled_children = [e for e in v_in.edges if not e.handled and e is not edge_in]
        if len(unhandled_children) >= 1:
            v_in.ref_twist = np.zeros(len(unhandled_children))

        edge_out.handled = True

        # Recurse
        tree_getKappa(edge_out, v_next, vertexObjs, edgeObjs)




