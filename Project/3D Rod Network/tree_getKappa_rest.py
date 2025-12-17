from computeSpaceParallel_OO import computeSpaceParallel_OO
from computeMaterialDirectors_OO import computeMaterialDirectors_OO
from PlotRodNetwork import PlotRodNetwork
from getKappa_OO import getKappa_OO
import numpy as np

def tree_getKappa_rest(edge_in, v_in, vertexObjs, edgeObjs):


    """
    Recursive curvature pass with junction handling.

    edge_in : the edge we arrived on (root/incoming)
    v_in    : the vertex we arrived at
    """
    edge_in.handled = True

    # --- Handle junction curvature ---
    if v_in.junction:
        v_in.junction_rest_kappa = []
        v_in.junction_flip = []

        def geom_tangent_away(edge, v):
            # Get tangent pointing away from vertex v
            other = edge.get_other_vertex(v)
            t = np.array(other.coords) - np.array(v.coords)
            return t / np.linalg.norm(t)

        # Loop over all pairs of edges at junction
        for edge0, edge1 in v_in.edgePairs:

            # Get the "other" vertices for each edge
            v0 = edge0.get_other_vertex(v_in)
            v2 = edge1.get_other_vertex(v_in)

            t0_away = geom_tangent_away(edge0, v_in)
            t1_away = geom_tangent_away(edge1, v_in)

            flip0 = True
            flip1 = False

            t0 = -t0_away if flip0 else t0_away
            t1 = -t1_away if flip1 else t1_away

            # --- Construct virtual nodes for curvature ---
            L = 0.5 * (edge0.rest_length + edge1.rest_length)  # or Voronoi length
            node0 = np.array(v_in.coords) - L * t0
            node1 = np.array(v_in.coords)
            node2 = np.array(v_in.coords) + L * t1

            # --- Compute rest curvature  ---
            kappaBar = getKappa_OO(node0, node1, node2, edge0, edge1)

            # Store results and flipped edges in junction
            v_in.junction_rest_kappa.append(kappaBar)
            v_in.junction_flip0.append(flip0)
            v_in.junction_flip1.append(flip1)

    # --- Normal recursive pass for child edges ---
    for edge_out in v_in.edges:
        if edge_out.handled or edge_out is edge_in:
            continue

        v_next = edge_out.get_other_vertex(v_in)
        v_prev = edge_in.get_other_vertex(v_in)

        # Compute curvature along the current edge
        v_in.rest_kappa = getKappa_OO(np.array(v_prev.coords), np.array(v_in.coords), np.array(v_next.coords), edge_in, edge_out)

        # Track tree structure
        edge_in.children.append(edge_out)
        edge_out.parent = edge_in

        # Allocate ref_twist if branching
        unhandled_children = [e for e in v_in.edges if not e.handled and e is not edge_in]
        if len(unhandled_children) >= 1:
            v_in.ref_twist = np.zeros(len(unhandled_children))

        edge_out.handled = True

        # Recurse
        tree_getKappa_rest(edge_out, v_next, vertexObjs, edgeObjs)