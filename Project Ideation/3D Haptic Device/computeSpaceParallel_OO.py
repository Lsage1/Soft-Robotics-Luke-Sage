import numpy as np
from computeTangent import computeTangent
from parallel_transport import parallel_transport

def computeSpaceParallel_OO(end_vertex, end_edge, edgeObjs, vertexObjs):
  # u1_first = first reference frame vector (arbitrary but orthonormal adapted) on the first edge
  # q is the DOF vector of size 4*nv - 1


  # Now that we have material directors for the end element, lets iterate over edges until we reach an end or a junction
  prev_edge = end_edge
  prev_vertex = end_vertex
  active = True
  while active:
    # Get the next edge in the chain
    active_vertex = prev_edge.get_other_vertex(prev_vertex)
    if active_vertex.end or active_vertex.junction:
        active = False
        break

    active_edge = active_vertex.get_other_edge(prev_edge)

    t0 = prev_edge.tangent
    t1 = active_edge.tangent

    active_edge.u1 = parallel_transport(prev_edge.u1, t0, t1)
    active_edge.u1 = active_edge.u1 / np.linalg.norm(active_edge.u1) # Ensure it is unit
    active_edge.u2 = np.cross(t1, active_edge.u1)
    active_edge.u2 = active_edge.u2 / np.linalg.norm(active_edge.u2) # Ensure it is unit

    prev_vertex = active_vertex
    prev_edge = active_edge
  return