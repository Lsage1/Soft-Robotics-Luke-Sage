import numpy as np

def GetRestDimensions(vertices, edgeIndex):
  # q - DOF vector of size N
  # EI - bending stiffness
  # deltaL - undeformed Voronoi length (assume to be a scalar for this simple example)
  # Output:
  # Fb - a vector (negative gradient of elastic stretching force)
  # Jb - a matrix (negative hessian of elastic stretching force)

  num_vertices = vertices.shape[0]
  vertex_edges = [[] for _ in range(num_vertices)]

  for eidx, edge in enumerate(edgeIndex):
    v0, v1 = edge
    vertex_edges[v0].append(eidx)
    vertex_edges[v1].append(eidx)

  for vertex, vertex_edge in enumerate(vertex_edges):
      print(vertex, vertex_edge)
      for i in vertex_edge:
          print(i)

  for i, edges in enumerate(vertex_edges):
      print(f"Vertex {i} is part of edges: {edges}")


  return


