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
    Recursive curvature pass.
    edge_in : the edge we arrived on
    v_in    : the vertex we arrived at
    """
    edge_in.handled = True

    # Iterate over all outgoing edges from current vertex
    for edge_out in v_in.edges:
        if edge_out.handled or edge_out is edge_in:
            continue  # Skip edges we've already handled or the incoming edge

        # Determine next vertex along this edge
        v_next = edge_out.get_other_vertex(v_in)
        v_prev = edge_in.get_other_vertex(v_in)  # previous vertex for curvature


        # Compute curvature
        getKappa_OO(v_prev, v_in, v_next, edge_in, edge_out)
        v_in.rest_kappa = v_in.kappa

        # Add Children / Parents
        edge_in.children.append(edge_out)
        edge_out.parent = edge_in
        print("assigned Children")
        print(edge_out.children)
        # Allocate ref_twist if branching
        unhandled_children = [e for e in v_in.edges if not e.handled and e is not edge_in]
        if len(unhandled_children) >= 1:
            v_in.ref_twist = np.zeros(len(unhandled_children))

        edge_out.handled = True

        # Optional visualization
        #PlotRodNetwork(vertexObjs, edgeObjs, extra_vertices=None, extra_edges=None)

        # Recurse forward
        tree_getKappa(edge_out, v_next, vertexObjs, edgeObjs)



