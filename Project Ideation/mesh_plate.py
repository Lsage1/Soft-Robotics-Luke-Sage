import pygmsh
import meshio
import numpy as np
import matplotlib.pyplot as plt

def mesh_plate(l, w, max_mesh_size):
    with pygmsh.geo.Geometry() as geom:
        # Define the rectangle geometry
        geom.add_rectangle(0, l, 0, w, 0, mesh_size=max_mesh_size)

        # Generate the mesh
        mesh = geom.generate_mesh()

    # Extract the node coordinates and elements
    nodes = mesh.points[:, :2]
    elements = mesh.cells_dict["triangle"]

    return nodes, elements

def plot_mesh(nodes, elements):
    plt.figure(figsize=(8, 6))

    # Plot each triangle element
    for element in elements:
        triangle = nodes[element, :]
        # Draw lines for each edge of the triangle
        for i in range(3):
            x_values = triangle[[i, (i + 1) % 3], 0]  # Connect vertices i and (i+1)%3
            y_values = triangle[[i, (i + 1) % 3], 1]
            plt.plot(x_values, y_values, color='black')  # Draw triangle edge

    # Plot nodes
    plt.scatter(nodes[:, 0], nodes[:, 1], color='red', label='Nodes', zorder=5)

    # Set plot properties
    plt.title('Mesh Visualization')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()

# Define your parameters
l = 1.0  # Length of the rectangle
w = 0.5  # Width of the rectangle
max_mesh_size = 0.1

# Generate mesh
nodes, elements = mesh_plate(l, w, max_mesh_size)

# Plot the mesh
plot_mesh(nodes, elements)