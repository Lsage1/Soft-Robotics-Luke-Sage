import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def diagnose_junction_bending(vertexObjs, edgeObjs):
    """
    Diagnostic tool to visualize junction bending configuration.
    Call this after tree_getKappa_rest to verify setup.
    """

    print("\n" + "=" * 80)
    print("JUNCTION BENDING DIAGNOSTIC")
    print("=" * 80)

    for v in vertexObjs:
        if not v.junction:
            continue

        print(f"\n{'=' * 60}")
        print(f"Junction at Vertex {v.id}: {v.coords}")
        print(f"{'=' * 60}")
        print(f"Connected edges: {[e.id for e in v.edges]}")
        print(f"Number of edge pairs: {len(v.edgePairs)}")

        # Check data consistency
        assert len(v.junction_rest_kappa) == len(v.edgePairs), \
            f"Mismatch: {len(v.junction_rest_kappa)} kappas vs {len(v.edgePairs)} pairs"
        assert len(v.junction_flip0) == len(v.edgePairs), \
            f"Mismatch: {len(v.junction_flip0)} flip0 vs {len(v.edgePairs)} pairs"
        assert len(v.junction_flip1) == len(v.edgePairs), \
            f"Mismatch: {len(v.junction_flip1)} flip1 vs {len(v.edgePairs)} pairs"

        # Visualize each pair
        fig = plt.figure(figsize=(15, 5 * len(v.edgePairs)))

        for i, (edge0, edge1) in enumerate(v.edgePairs):
            print(f"\n  Pair {i}: Edges ({edge0.id}, {edge1.id})")
            print(f"    flip0={v.junction_flip0[i]}, flip1={v.junction_flip1[i]}")
            print(f"    Rest kappa: {v.junction_rest_kappa[i]}")

            # Get vertices
            v0 = edge0.get_other_vertex(v)
            v1 = v
            v2 = edge1.get_other_vertex(v)

            print(f"    Vertices: {v0.id} → {v1.id} → {v2.id}")
            print(f"    Positions: {v0.coords} → {v1.coords} → {v2.coords}")

            # Compute tangents
            def geom_tangent_away(edge, vertex):
                other = edge.get_other_vertex(vertex)
                t = np.array(other.coords) - np.array(vertex.coords)
                return t / np.linalg.norm(t)

            t0_original = geom_tangent_away(edge0, v)
            t1_original = geom_tangent_away(edge1, v)

            t0 = -t0_original if v.junction_flip0[i] else t0_original
            t1 = -t1_original if v.junction_flip1[i] else t1_original

            print(f"    t0 original: {t0_original}")
            print(f"    t0 used:     {t0}")
            print(f"    t1 original: {t1_original}")
            print(f"    t1 used:     {t1}")

            # Compute virtual nodes
            L = 0.5 * (edge0.rest_length + edge1.rest_length)
            node0 = np.array(v.coords) - L * t0
            node1 = np.array(v.coords)
            node2 = np.array(v.coords) + L * t1

            print(f"    Virtual node0: {node0}")
            print(f"    Virtual node1: {node1} (junction)")
            print(f"    Virtual node2: {node2}")

            # Check if virtual nodes make geometric sense
            dist0 = np.linalg.norm(node0 - np.array(v0.coords))
            dist2 = np.linalg.norm(node2 - np.array(v2.coords))
            print(f"    Distance node0 to vertex_{v0.id}: {dist0:.4f}")
            print(f"    Distance node2 to vertex_{v2.id}: {dist2:.4f}")

            if dist0 > edge0.rest_length or dist2 > edge1.rest_length:
                print(f"    ⚠️  WARNING: Virtual nodes are far from actual vertices!")

            # Plot
            ax = fig.add_subplot(len(v.edgePairs), 1, i + 1, projection='3d')

            # Plot actual edges
            for e in v.edges:
                p1 = np.array(e.vertex1.coords)
                p2 = np.array(e.vertex2.coords)
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        'b-', linewidth=2, alpha=0.3, label='Other edges' if e not in [edge0, edge1] else '')

            # Highlight the pair
            p0 = np.array(v0.coords)
            p1 = np.array(v1.coords)
            p2 = np.array(v2.coords)

            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    'r-', linewidth=3, label=f'Edge {edge0.id}')
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    'g-', linewidth=3, label=f'Edge {edge1.id}')

            # Plot virtual nodes
            ax.scatter(*node0, c='red', s=100, marker='s', label='Virtual node0')
            ax.scatter(*node1, c='yellow', s=100, marker='o', label='Junction')
            ax.scatter(*node2, c='green', s=100, marker='s', label='Virtual node2')

            # Plot virtual edges
            ax.plot([node0[0], node1[0]], [node0[1], node1[1]], [node0[2], node1[2]],
                    'r--', linewidth=2, alpha=0.5)
            ax.plot([node1[0], node2[0]], [node1[1], node2[1]], [node1[2], node2[2]],
                    'g--', linewidth=2, alpha=0.5)

            # Plot tangent arrows
            arrow_scale = L * 0.3
            ax.quiver(*p1, *t0 * arrow_scale, color='red', arrow_length_ratio=0.3, linewidth=2)
            ax.quiver(*p1, *t1 * arrow_scale, color='green', arrow_length_ratio=0.3, linewidth=2)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            ax.set_title(f'Junction {v.id}, Pair {i}: Edges ({edge0.id},{edge1.id})')

            # Set equal aspect ratio
            all_points = np.array([p0, p1, p2, node0, node2])
            max_range = np.array([all_points[:, i].max() - all_points[:, i].min() for i in range(3)]).max() / 2.0
            mid = all_points.mean(axis=0)
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        plt.tight_layout()
        plt.show()

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80 + "\n")

# Add this to your main simulation code after tree_getKappa_rest:
# diagnose_junction_bending(vertexObjs, edgeObjs)