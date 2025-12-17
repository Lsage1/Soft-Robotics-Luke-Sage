import numpy as np


def detect_junction_flip(vertexObjs, edgeObjs, timestep):
    """
    Detect if junction geometry has flipped relative to rest configuration.

    The key indicator: the sign of the curvature binormal should remain consistent.
    If it flips, the junction has inverted.
    """

    print(f"\n{'=' * 80}")
    print(f"JUNCTION FLIP DETECTION - Timestep {timestep}")
    print(f"{'=' * 80}")

    for vertex in vertexObjs:
        if not vertex.junction:
            continue

        print(f"\n--- Junction Vertex {vertex.id} ---")

        # Store initial chirality if not already stored
        if not hasattr(vertex, 'rest_chirality'):
            vertex.rest_chirality = []

        for i, (edge0, edge1) in enumerate(vertex.edgePairs):
            v0 = edge0.get_other_vertex(vertex)
            v2 = edge1.get_other_vertex(vertex)

            # Current configuration
            node0_curr = np.array(v0.coords)
            node1_curr = np.array(vertex.coords)
            node2_curr = np.array(v2.coords)

            # Rest configuration
            node0_rest = np.array(v0.rest_coords)
            node1_rest = np.array(vertex.rest_coords)
            node2_rest = np.array(v2.rest_coords)

            # Compute edge vectors
            e0_curr = node0_curr - node1_curr
            e2_curr = node2_curr - node1_curr

            e0_rest = node0_rest - node1_rest
            e2_rest = node2_rest - node1_rest

            # Compute normal to the plane formed by the two edges
            # This indicates the "handedness" of the junction
            normal_curr = np.cross(e0_curr, e2_curr)
            normal_rest = np.cross(e0_rest, e2_rest)

            # Normalize
            if np.linalg.norm(normal_curr) > 1e-10:
                normal_curr = normal_curr / np.linalg.norm(normal_curr)
            if np.linalg.norm(normal_rest) > 1e-10:
                normal_rest = normal_rest / np.linalg.norm(normal_rest)

            print(f"\nEdge Pair {i} (Edge {edge0.id} & Edge {edge1.id}):")
            print(f"  Normal (rest):    {normal_rest}")
            print(f"  Normal (current): {normal_curr}")

            # Check if normal has flipped
            normal_dot = np.dot(normal_curr, normal_rest)
            print(f"  Dot product: {normal_dot:.6f}")

            if normal_dot < 0:
                print(f"  ⚠️  JUNCTION FLIPPED! Normal points opposite direction!")
                print(f"     This means the junction has inverted spatially.")
                print(f"     Curvature sign will be wrong, causing phantom forces!")
            elif normal_dot < 0.5:
                print(f"  ⚠️  WARNING: Junction geometry changed significantly!")
            else:
                print(f"  ✓ Junction orientation consistent with rest")

            # Also check the angle between edges
            e0_curr_norm = e0_curr / np.linalg.norm(e0_curr)
            e2_curr_norm = e2_curr / np.linalg.norm(e2_curr)
            e0_rest_norm = e0_rest / np.linalg.norm(e0_rest)
            e2_rest_norm = e2_rest / np.linalg.norm(e2_rest)

            angle_curr = np.arccos(np.clip(np.dot(e0_curr_norm, e2_curr_norm), -1, 1))
            angle_rest = np.arccos(np.clip(np.dot(e0_rest_norm, e2_rest_norm), -1, 1))

            print(f"  Angle between edges:")
            print(f"    Rest:    {np.degrees(angle_rest):.2f}°")
            print(f"    Current: {np.degrees(angle_curr):.2f}°")
            print(f"    Change:  {np.degrees(angle_curr - angle_rest):.2f}°")


def compute_signed_curvature_with_flip_detection(vertex, edge0, edge1, pair_idx):
    """
    Compute curvature with automatic flip detection and correction.
    """

    v0 = edge0.get_other_vertex(vertex)
    v2 = edge1.get_other_vertex(vertex)

    node0 = np.array(v0.coords)
    node1 = np.array(vertex.coords)
    node2 = np.array(v2.coords)

    # Compute tangents
    ee = node1 - node0
    ef = node2 - node1
    te = ee / np.linalg.norm(ee)
    tf = ef / np.linalg.norm(ef)

    # Compute curvature binormal
    kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf))

    # Check if binormal has flipped relative to rest
    if not hasattr(vertex, 'rest_binormals'):
        # Initialize rest binormals
        vertex.rest_binormals = []
        for i, (e0, e1) in enumerate(vertex.edgePairs):
            v0_r = e0.get_other_vertex(vertex)
            v2_r = e1.get_other_vertex(vertex)
            n0 = np.array(v0_r.rest_coords)
            n1 = np.array(vertex.rest_coords)
            n2 = np.array(v2_r.rest_coords)
            ee_r = n1 - n0
            ef_r = n2 - n1
            te_r = ee_r / np.linalg.norm(ee_r)
            tf_r = ef_r / np.linalg.norm(ef_r)
            kb_r = 2.0 * np.cross(te_r, tf_r) / (1.0 + np.dot(te_r, tf_r))
            vertex.rest_binormals.append(kb_r)

    kb_rest = vertex.rest_binormals[pair_idx]

    # Check if binormal flipped
    flip_sign = 1.0
    if np.linalg.norm(kb) > 1e-10 and np.linalg.norm(kb_rest) > 1e-10:
        kb_dot = np.dot(kb / np.linalg.norm(kb), kb_rest / np.linalg.norm(kb_rest))
        if kb_dot < 0:
            flip_sign = -1.0
            print(f"  Detected binormal flip for pair {pair_idx}, applying correction")

    return flip_sign


# Usage:
"""
if timeStep >= 165 and timeStep <= 175:
    detect_junction_flip(vertexObjs, edgeObjs, timeStep)
"""