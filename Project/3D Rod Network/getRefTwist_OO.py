import numpy as np
from computeReferenceTwist import computeReferenceTwist
from PlotRodNetwork import PlotRodNetwork

def getRefTwist_OO(vertexObjs, edgeObjs, end_edge, first_end_vertex):

    # Given all the reference frames along the rod, we calculate the reference
    # twist along the rod on every node.
    # Will calculate referenceTwist for all internal nodes (all nodes except terminal nodes


    for edge in edgeObjs:
        edge.handled = False
    activeJunction = []
    tree_search_active = True
    edge01 = end_edge
    vertex0 = first_end_vertex
    print("GettingRefTwist")

    while tree_search_active:
        vertex1 = edge01.get_other_vertex(vertex0)
        # Only add junction if it hasn't been added yet
        if vertex1.junction:
            # Get all outgoing edges except the one we came from
            remaining = [e for e in vertex1.edges if (not e.handled and e is not edge01)]

            # Only push the junction if it has future branches
            if len(remaining) > 0 and all(j[0] != vertex1 for j in activeJunction):
                activeJunction.append([vertex1, edge01])

        edge12, found_edge = vertex1.get_unhandled_edge(edge01)

        if not found_edge:
            if len(activeJunction) > 0:
                #print("hit an end!")
                # Backtrack through junctions until we find one with unhandled edges
                while len(activeJunction) > 0:
                    vertex1, edge01 = activeJunction[-1]  # Most recent junction
                    edge12, found_edge = vertex1.get_unhandled_edge(edge01)
                    if found_edge:
                        vertex0 = edge01.get_other_vertex(vertex1)
                        break  # Found an edge, exit backtracking
                    else:
                        activeJunction.pop()  # No edges left at this junction, remove it
                else:
                    # If we exhausted all junctions
                    #print("No edge found! Assuming tree is finished")
                    break

            else:
                print("No edge found! Assuming tree is finished")
                break

        PlotRodNetwork(vertexObjs, edgeObjs, [], [edge01, edge12])

        # Make sure every edge knows the edge it came from. This will be used to calculate twisting between beams
        edge12.root = edge01
        vertex2 = edge12.get_other_vertex(vertex1)

        # Compute reference twist:
        a1e = edge01.a1
        a1f = edge12.a1
        t1 = edge01.tangent
        t2 = edge12.tangent
        child_index = edge12.get_branch_number()
        print("vertex1 reference twist", vertex1.ref_twist, edge01.children)
        vertex1.ref_twist[child_index] = computeReferenceTwist(a1e, a1f, t1, t2, vertex1.ref_twist[child_index])

        # Move on to next edge and vertex
        edge12.handled = True
        edge01 = edge12
        vertex0 = vertex1
        vertex1 = vertex2