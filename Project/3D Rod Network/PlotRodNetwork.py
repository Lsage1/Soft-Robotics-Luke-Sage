import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def PlotRodNetwork(vertexObjs,
                   edgeObjs,
                   extra_vertices=None,
                   extra_edges=None):
    """
    Plot a rod network with fixed world scaling.
    The axis limits are computed ONCE (first call) and then reused.
    """

    # ============================================================
    #                    VISUAL SETTINGS
    # ============================================================
    director_scale = 0.15
    tangent_scale  = 0.25

    show_directors = True
    show_edge_directions = True

    # ============================================================
    #              ONE-TIME AUTO SCALE (CACHED)
    # ============================================================
    if not hasattr(PlotRodNetwork, "_axis_limits"):
        all_coords = np.array([v.coords for v in vertexObjs], dtype=float)

        center = all_coords.mean(axis=0)
        half_range = 2 * np.ptp(all_coords, axis=0).max()

        # Robust fallback for very small systems
        if half_range < 1e-6:
            half_range = 1.0

        PlotRodNetwork._axis_limits = {
            "x": (center[0] - half_range, center[0] + half_range),
            "y": (center[1] - half_range, center[1] + half_range),
            "z": (center[2] - half_range, center[2] + half_range),
        }

    lims = PlotRodNetwork._axis_limits

    # ============================================================
    #                       FIGURE
    # ============================================================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # ============================================================
    #                        VERTICES
    # ============================================================
    xs = [v.coords[0] for v in vertexObjs]
    ys = [v.coords[1] for v in vertexObjs]
    zs = [v.coords[2] for v in vertexObjs]

    ax.scatter(xs, ys, zs, s=40, color="black")

    # ============================================================
    #                          EDGES
    # ============================================================
    for e in edgeObjs:
        p1 = np.asarray(e.vertex1.coords, dtype=float)
        p2 = np.asarray(e.vertex2.coords, dtype=float)

        # Edge geometry
        ax.plot([p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color="blue")

        # Highlight handled edges
        if getattr(e, "handled", False):
            ax.plot([p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color="green",
                    linewidth=6)

        mid = 0.5 * (p1 + p2)

        # --------------------------------------------------------
        # Tangent direction
        # --------------------------------------------------------
        if show_edge_directions and hasattr(e, "tangent"):
            t = np.asarray(e.tangent, dtype=float)
            t_norm = np.linalg.norm(t)

            if t_norm > 1e-12:
                t_hat = t / t_norm

                ax.quiver(mid[0], mid[1], mid[2],
                          t_hat[0], t_hat[1], t_hat[2],
                          length=tangent_scale,
                          normalize=True,
                          color="black",
                          linewidth=1.5)

        # --------------------------------------------------------
        # Material directors
        # --------------------------------------------------------
        if show_directors and hasattr(e, "m1") and hasattr(e, "m2"):
            m1 = np.asarray(e.m1, dtype=float)
            m2 = np.asarray(e.m2, dtype=float)

            ax.quiver(mid[0], mid[1], mid[2],
                      m1[0], m1[1], m1[2],
                      length=director_scale,
                      normalize=True,
                      color="red")

            ax.quiver(mid[0], mid[1], mid[2],
                      m2[0], m2[1], m2[2],
                      length=director_scale,
                      normalize=True,
                      color="orange")

    # ============================================================
    #                EXTRA VERTICES / EDGES
    # ============================================================
    if extra_vertices is not None:
        colors = ["red", "green", "blue"]
        for v, c in zip(extra_vertices, colors):
            ax.scatter(v.coords[0], v.coords[1], v.coords[2],
                       s=80, color=c, edgecolor="k")

    if extra_edges is not None:
        colors = ["magenta", "orange"]
        for e, c in zip(extra_edges, colors):
            p1 = e.vertex1.coords
            p2 = e.vertex2.coords
            ax.plot([p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color=c,
                    linewidth=3)

    # ============================================================
    #                   FIXED AXIS SCALING
    # ============================================================
    ax.set_xlim(*lims["x"])
    ax.set_ylim(*lims["y"])
    ax.set_zlim(*lims["z"])

    ax.set_box_aspect((
        lims["x"][1] - lims["x"][0],
        lims["y"][1] - lims["y"][0],
        lims["z"][1] - lims["z"][0],
    ))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()

