import numpy as np
from computeTangent import computeTangent
from parallel_transport import parallel_transport
from ComputeTangentEdges import ComputeTangentEdges

def computeTimeParallel_OO(end_edge, first_end_vertex, edgeObjs, q0, q):
  # a1_old: First time parallel frame director in "old" configuration
  # q0: "old" shape of the rod or DOF vector
  # q: "new" shape (a1 on this new shape is unknown)

  nv = (len(q)+1) // 4
  ne = nv -1


  ComputeTangentEdges(edgeObjs, False)
  ComputeTangentEdges(edgeObjs, True)

  a1 = end_edge.a1 # First time parallel frame director
  a2 = end_edge.a2 # Second time parallel frame director

  t0 = end_edge.tangent0  # old tangent on the c-th edge
  t1 = end_edge.tangent  # new tangent on the c-th edge
  end_edge.a1 = parallel_transport(end_edge.a1_old, t0, t1)
  end_edge.a1 = end_edge.a1 - np.dot(end_edge.a1, t1) * t1  # Ensure it is orthogonal to t1
  end_edge.a1 = end_edge.a1 / np.linalg.norm(end_edge.a1)  # Ensure it is unit
  end_edge.a2 = np.cross(t1, end_edge.a1)
  end_edge.a2 = end_edge.a2 - np.dot(end_edge.a2, t1) * t1  # Ensure it is orthogonal to t1
  end_edge.a2 = end_edge.a2 / np.linalg.norm(end_edge.a2)  # Ensure it is unit
  end_edge.handled = True

# Another tree search!
  tree_search_active = True
  edge01 = end_edge
  vertex0 = first_end_vertex
  activeJunction = []  # When a junction is reached, store the junction vertex, and the root edge that led into it
  for edge in edgeObjs:
      edge.handled = False
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
                  break
          else:
              break

      # Make sure every edge knows the edge it came from. This will be used to calculate twisting between beams
      edge12.root = edge01
      vertex2 = edge12.get_other_vertex(vertex1)

      ######################################
      t0 = edge12.tangent0  # old tangent on the c-th edge
      t1 = edge12.tangent  # new tangent on the c-th edge
      edge12.a1 = parallel_transport(edge12.a1_old, t0, t1)
      edge12.a1 = edge12.a1 - np.dot(edge12.a1, t1) * t1  # Ensure it is orthogonal to t1
      edge12.a1 = edge12.a1 / np.linalg.norm(edge12.a1)  # Ensure it is unit
      edge12.a2 = np.cross(t1, edge12.a1)
      edge12.a2 = edge12.a2 - np.dot(edge12.a2, t1) * t1  # Ensure it is orthogonal to t1
      edge12.a2 = edge12.a2 / np.linalg.norm(edge12.a2)  # Ensure it is unit
      ######################################

      # Move on to next edge and vertex
      edge12.handled = True
      edge01 = edge12
      vertex0 = vertex1
      vertex1 = vertex2

  return