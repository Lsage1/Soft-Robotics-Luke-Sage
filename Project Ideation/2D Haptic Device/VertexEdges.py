import numpy as np

def VertexEdges(vertices, edgeIndex):

    num_vertices = vertices.shape[0]
    vertex_edges = [[] for _ in range(num_vertices)]

    for edge_idx, edge in enumerate(edgeIndex):
        v0, v1 = edge
        vertex_edges[v0].append(edge_idx)
        vertex_edges[v1].append(edge_idx)

    return vertex_edges

# Example usage
vertices = np.array([[0,0], [0.02,0], [.04,0], [.06, 0], [.08, 0], [.1, 0], [0.05, -0.05]])
edgeIndex = np.array([[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [0,6], [3,6]])

vertex_edges = edges_per_vertex(vertices, edgeIndex)

for i, edges in enumerate(vertex_edges):
    print(f"Vertex {i} is part of edges: {edges}")