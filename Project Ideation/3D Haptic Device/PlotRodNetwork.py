import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def PlotRodNetwork(vertexObjs, edgeObjs, extra_vertices=None, extra_edges=None):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    # ---- Original Vertex Plotting ----
    xs = [v.coords[0] for v in vertexObjs]
    ys = [v.coords[1] for v in vertexObjs]
    zs = [v.coords[2] for v in vertexObjs]
    ax.scatter(xs, ys, zs, s=40, color='black')

    # ---- Original Edge Plotting ----
    for e in edgeObjs:
        p1 = e.vertex1.coords
        p2 = e.vertex2.coords
        xs = [p1[0], p2[0]]
        ys = [p1[1], p2[1]]
        zs = [p1[2], p2[2]]

        ax.plot(xs, ys, zs, color='blue')

        # If handled, overlay in red
        if getattr(e, 'handled', False):
            ax.plot(xs, ys, zs, color='red', linewidth=3)

    # ============================================================
    #              OVERLAY: SPECIAL VERTICES & EDGES
    # ============================================================

    # ---- Extra Vertices (3 vertices) ----
    if extra_vertices is not None:
        vert_colors = ['red', 'green', 'blue']  # 3 distinct colors
        for v, c in zip(extra_vertices, vert_colors):
            ax.scatter(v.coords[0], v.coords[1], v.coords[2],
                       s=80, color=c, edgecolor='k', linewidth=1)

    # ---- Extra Edges (2 edges) ----
    if extra_edges is not None:
        edge_colors = ['magenta', 'orange']  # 2 distinct colors
        for e, c in zip(extra_edges, edge_colors):
            p1 = e.vertex1.coords
            p2 = e.vertex2.coords
            xs = [p1[0], p2[0]]
            ys = [p1[1], p2[1]]
            zs = [p1[2], p2[2]]

            ax.plot(xs, ys, zs, color=c, linewidth=3)

    # ---- Equal axis scaling ----
    all_coords = [v.coords for v in vertexObjs]
    all_coords = list(zip(*all_coords))

    max_range = max(
        max(all_coords[0]) - min(all_coords[0]),
        max(all_coords[1]) - min(all_coords[1]),
        max(all_coords[2]) - min(all_coords[2])
    ) / 2.0

    mid = [sum(c)/len(c) for c in all_coords]
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
