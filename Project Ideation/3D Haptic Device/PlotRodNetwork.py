import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def PlotRodNetwork(vertexObjs, edgeObjs):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    # Plot vertices
    xs = [v.coords[0] for v in vertexObjs]
    ys = [v.coords[1] for v in vertexObjs]
    zs = [v.coords[2] for v in vertexObjs]
    ax.scatter(xs, ys, zs, s=40)

    # Plot edges
    for e in edgeObjs:
        p1 = e.vertex1.coords
        p2 = e.vertex2.coords
        xs = [p1[0], p2[0]]
        ys = [p1[1], p2[1]]
        zs = [p1[2], p2[2]]

        # Plot every edge in blue
        ax.plot(xs, ys, zs, color='blue')

        # If edge is handled, also plot in red
        if getattr(e, 'handled', False):
            ax.plot(xs, ys, zs, color='red')

    # Make axes equal scale
    all_coords = [v.coords for v in vertexObjs]
    all_coords = list(zip(*all_coords))
    max_range = max(max(all_coords[0]) - min(all_coords[0]),
                    max(all_coords[1]) - min(all_coords[1]),
                    max(all_coords[2]) - min(all_coords[2])) / 2.0

    mid = [sum(c)/len(c) for c in all_coords]
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
