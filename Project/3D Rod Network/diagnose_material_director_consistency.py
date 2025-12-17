import numpy as np
from computeKappa import computeKappa


def diagnose_material_director_consistency(vertexObjs, edgeObjs, EI):
    """
    Check if material directors are causing the force explosion.

    Key checks:
    1. Are m1, m2 orthogonal to tangent?
    2. Are m1, m2 orthogonal to each other?
    3. Are m1, m2 unit length?
    4. Do material directors match between rest and current for similar geometry?
    """

    print("\n" + "=" * 80)
    print("MATERIAL DIRECTOR CONSISTENCY CHECK")
    print("=" * 80)

    problems_found = False

    for edge in edgeObjs:
        print(f"\n--- Edge {edge.id} (v{edge.vertex1.id} → v{edge.vertex2.id}) ---")

        t = edge.tangent
        m1 = edge.m1
        m2 = edge.m2

        # Check 1: Orthogonality to tangent
        m1_dot_t = np.dot(m1, t)
        m2_dot_t = np.dot(m2, t)

        if abs(m1_dot_t) > 0.01:
            print(f"  ⚠️  m1 NOT orthogonal to tangent! m1·t = {m1_dot_t:.6f}")
            problems_found = True

        if abs(m2_dot_t) > 0.01:
            print(f"  ⚠️  m2 NOT orthogonal to tangent! m2·t = {m2_dot_t:.6f}")
            problems_found = True

        # Check 2: Orthogonality to each other
        m1_dot_m2 = np.dot(m1, m2)
        if abs(m1_dot_m2) > 0.01:
            print(f"  ⚠️  m1, m2 NOT orthogonal! m1·m2 = {m1_dot_m2:.6f}")
            problems_found = True

        # Check 3: Unit length
        m1_norm = np.linalg.norm(m1)
        m2_norm = np.linalg.norm(m2)

        if abs(m1_norm - 1.0) > 0.01:
            print(f"  ⚠️  m1 NOT unit length! |m1| = {m1_norm:.6f}")
            problems_found = True

        if abs(m2_norm - 1.0) > 0.01:
            print(f"  ⚠️  m2 NOT unit length! |m2| = {m2_norm:.6f}")
            problems_found = True

        # Check 4: Theta value
        print(f"  theta = {edge.theta:.4f} rad ({np.degrees(edge.theta):.1f}°)")

        if not problems_found:
            print(f"  ✓ Material directors OK")

    return problems_found


def diagnose_curvature_at_vertex(vertex, edgeObjs, timestep, EI):
    """
    Detailed diagnosis of curvature calculation at a specific vertex.
    Shows exactly what goes into gradEb_hessEb.
    """

    print(f"\n" + "=" * 80)
    print(f"DETAILED CURVATURE ANALYSIS - Vertex {vertex.id} at timestep {timestep}")
    print("=" * 80)

    if vertex.junction:
        print(f"Junction with {len(vertex.edgePairs)} edge pairs")

        for i, (edge0, edge1) in enumerate(vertex.edgePairs):
            print(f"\n--- Edge Pair {i}: Edge {edge0.id} & Edge {edge1.id} ---")

            v0 = edge0.get_other_vertex(vertex)
            v2 = edge1.get_other_vertex(vertex)

            node0 = np.array(v0.coords)
            node1 = np.array(vertex.coords)
            node2 = np.array(v2.coords)

            print(f"Node positions:")
            print(f"  node0 (v{v0.id}): {node0}")
            print(f"  node1 (v{vertex.id}): {node1}")
            print(f"  node2 (v{v2.id}): {node2}")

            # Compute tangents from geometry (what gradEb_hessEb does)
            ee = node1 - node0
            ef = node2 - node1
            te = ee / np.linalg.norm(ee)
            tf = ef / np.linalg.norm(ef)

            print(f"Tangents from geometry:")
            print(f"  te (edge0): {te}")
            print(f"  tf (edge1): {tf}")

            # Compare with stored tangents
            print(f"Stored tangents:")
            print(f"  edge0.tangent: {edge0.tangent}")
            print(f"  edge1.tangent: {edge1.tangent}")

            # Check if they match
            te_match = np.allclose(te, edge0.tangent) or np.allclose(te, -edge0.tangent)
            tf_match = np.allclose(tf, edge1.tangent) or np.allclose(tf, -edge1.tangent)

            if not te_match:
                print(f"  ⚠️  WARNING: te doesn't match edge0.tangent!")
            if not tf_match:
                print(f"  ⚠️  WARNING: tf doesn't match edge1.tangent!")

            # Material directors
            m1e = edge0.m1
            m2e = edge0.m2
            m1f = edge1.m1
            m2f = edge1.m2

            print(f"Material directors:")
            print(f"  edge0: m1={m1e}, m2={m2e}")
            print(f"  edge1: m1={m1f}, m2={m2f}")

            # Compute curvature binormal
            kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf))

            # Compute curvatures
            kappa1 = 0.5 * np.dot(kb, m2e + m2f)
            kappa2 = -0.5 * np.dot(kb, m1e + m1f)

            kappa_current = np.array([kappa1, kappa2])
            kappa_rest = np.array(vertex.junction_rest_kappa[i])

            print(f"Curvatures:")
            print(f"  kb (binormal): {kb}")
            print(f"  κ_current: {kappa_current}")
            print(f"  κ_rest:    {kappa_rest}")
            print(f"  Δκ:        {kappa_current - kappa_rest}")
            print(f"  |Δκ|:      {np.linalg.norm(kappa_current - kappa_rest):.6f}")

            # Estimate force magnitude
            dL = 0.5 * (edge0.rest_length + edge1.rest_length)
            force_est = EI * np.linalg.norm(kappa_current - kappa_rest) / dL
            print(f"  Estimated force magnitude: {force_est:.6e} N")

            if force_est > 1.0:
                print(f"  ⚠️  LARGE FORCE! This is likely causing the explosion!")

    elif not vertex.end and len(vertex.edges) == 2:
        print(f"Regular vertex with 2 edges")

        edge_in = vertex.edges[0]
        edge_out = vertex.edges[1]

        v0 = edge_in.get_other_vertex(vertex)
        v2 = edge_out.get_other_vertex(vertex)

        node0 = np.array(v0.coords)
        node1 = np.array(vertex.coords)
        node2 = np.array(v2.coords)

        print(f"Node positions:")
        print(f"  node0 (v{v0.id}): {node0}")
        print(f"  node1 (v{vertex.id}): {node1}")
        print(f"  node2 (v{v2.id}): {node2}")

        # Similar analysis as junction...
        kappa_current = vertex.kappa if hasattr(vertex, 'kappa') else None
        kappa_rest = vertex.rest_kappa

        if kappa_current:
            print(f"Curvatures:")
            print(f"  κ_current: {kappa_current}")
            print(f"  κ_rest:    {kappa_rest}")


def track_vertex_through_time(vertex, edgeObjs, history_data):
    """
    Track how a specific vertex's properties change over time.
    Call this every timestep to build history.

    history_data should be a dict that gets updated each timestep.
    """

    if not hasattr(vertex, 'id'):
        return

    v_id = vertex.id

    if v_id not in history_data:
        history_data[v_id] = {
            'coords': [],
            'junction_kappa': [],
            'material_dirs': []
        }

    history_data[v_id]['coords'].append(np.array(vertex.coords))

    if vertex.junction:
        kappa_list = []
        for i in range(len(vertex.edgePairs)):
            if hasattr(vertex, 'junction_kappa') and len(vertex.junction_kappa) > i:
                kappa_list.append(vertex.junction_kappa[i])
        history_data[v_id]['junction_kappa'].append(kappa_list)

    # Track material directors of adjacent edges
    for edge in vertex.edges:
        if not any(e['edge_id'] == edge.id for e in history_data[v_id]['material_dirs']):
            history_data[v_id]['material_dirs'].append({
                'edge_id': edge.id,
                'm1': [edge.m1.copy()],
                'm2': [edge.m2.copy()],
                'theta': [edge.theta]
            })
        else:
            for e_data in history_data[v_id]['material_dirs']:
                if e_data['edge_id'] == edge.id:
                    e_data['m1'].append(edge.m1.copy())
                    e_data['m2'].append(edge.m2.copy())
                    e_data['theta'].append(edge.theta)


# Usage:
"""
# In your main loop, right before forces explode:

if timeStep == 170:  # When you see problems starting
    diagnose_material_director_consistency(vertexObjs, edgeObjs, EI)

    # Focus on the vertex with largest force
    for vertex in vertexObjs:
        if vertex.junction:
            diagnose_curvature_at_vertex(vertex, edgeObjs, timeStep)

# Or track continuously:
history = {}
for timeStep in range(Nsteps):
    # ... your simulation ...

    for vertex in vertexObjs:
        track_vertex_through_time(vertex, edgeObjs, history)

    if timeStep == 185:  # After explosion
        # Analyze history to see what changed dramatically
        for v_id, data in history.items():
            coords = np.array(data['coords'])
            # Plot or analyze how coords changed over time
"""