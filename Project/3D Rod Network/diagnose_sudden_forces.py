import numpy as np


def diagnose_sudden_forces(vertexObjs, edgeObjs, Fb, Fs, timeStep, threshold=1e-2):
    """
    Diagnose sudden large forces in the simulation.
    Call this every timestep to catch issues early.

    Args:
        vertexObjs: list of vertex objects
        edgeObjs: list of edge objects
        Fb: bending force vector
        Fs: stretching force vector
        timeStep: current timestep number
        threshold: force magnitude threshold for warnings
    """

    print(f"\n=== Diagnostic for timestep {timeStep} ===")

    # 1. Check for material director flips
    print("\n1. MATERIAL DIRECTOR CONTINUITY:")
    for edge in edgeObjs:
        if hasattr(edge, 'm1_old'):
            m1_dot = np.dot(edge.m1, edge.m1_old)
            m2_dot = np.dot(edge.m2, edge.m2_old)

            if m1_dot < 0.7 or m2_dot < 0.7:
                print(f"  ⚠️  Edge {edge.id}: m1_dot={m1_dot:.4f}, m2_dot={m2_dot:.4f}")
                print(f"      Theta: {edge.theta:.4f}, Rest: {edge.rest_twist:.4f}")

        # Store for next iteration
        edge.m1_old = edge.m1.copy()
        edge.m2_old = edge.m2.copy()

    # 2. Check for tangent discontinuities
    print("\n2. TANGENT CONTINUITY:")
    for edge in edgeObjs:
        if hasattr(edge, 'tangent0'):
            tangent_dot = np.dot(edge.tangent, edge.tangent0)

            if abs(tangent_dot) < 0.95:
                print(f"  ⚠️  Edge {edge.id}: tangent_dot={tangent_dot:.4f}")
                print(f"      v1: {edge.vertex1.coords}")
                print(f"      v2: {edge.vertex2.coords}")

    # 3. Check for large curvature changes
    print("\n3. CURVATURE CHANGES:")
    for vertex in vertexObjs:
        if vertex.junction:
            for i, (kappa_rest) in enumerate(vertex.junction_rest_kappa):
                # Note: You need to compute current kappa and store it in vertex.junction_kappa
                if hasattr(vertex, 'junction_kappa') and len(vertex.junction_kappa) > i:
                    kappa_current = vertex.junction_kappa[i]
                    kappa_diff = np.array(kappa_current) - np.array(kappa_rest)
                    kappa_mag = np.linalg.norm(kappa_diff)

                    if kappa_mag > 5.0:  # Threshold for "large" curvature
                        print(f"  ⚠️  Junction vertex {vertex.id}, pair {i}:")
                        print(f"      Current kappa: {kappa_current}")
                        print(f"      Rest kappa: {kappa_rest}")
                        print(f"      Magnitude: {kappa_mag:.4f}")

        elif not vertex.end and vertex.rest_kappa[0] is not None:
            if hasattr(vertex, 'kappa'):
                kappa_diff = np.array(vertex.kappa) - np.array(vertex.rest_kappa)
                kappa_mag = np.linalg.norm(kappa_diff)

                if kappa_mag > 5.0:
                    print(f"  ⚠️  Regular vertex {vertex.id}:")
                    print(f"      Current kappa: {vertex.kappa}")
                    print(f"      Rest kappa: {vertex.rest_kappa}")
                    print(f"      Magnitude: {kappa_mag:.4f}")

    # 4. Check force magnitudes
    print("\n4. FORCE MAGNITUDES:")
    max_bending_force = np.max(np.abs(Fb))
    max_stretch_force = np.max(np.abs(Fs))

    print(f"  Max bending force: {max_bending_force:.6e}")
    print(f"  Max stretch force: {max_stretch_force:.6e}")

    if max_bending_force > threshold:
        # Find which DOF has the largest force
        max_idx = np.argmax(np.abs(Fb))
        print(f"  ⚠️  Large bending force at DOF {max_idx}: {Fb[max_idx]:.6e}")

        # Identify which vertex/edge this belongs to
        for vertex in vertexObjs:
            if max_idx in vertex.index:
                coord_idx = vertex.index.index(max_idx)
                print(f"      This is vertex {vertex.id}, coordinate {['x', 'y', 'z'][coord_idx]}")
                print(f"      Position: {vertex.coords}")
                break

        for edge in edgeObjs:
            if max_idx == edge.theta_index:
                print(f"      This is edge {edge.id}, theta DOF")
                print(f"      Theta: {edge.theta:.4f}")
                break

    # 5. Check for inverted elements (negative edge lengths)
    print("\n5. ELEMENT VALIDITY:")
    for edge in edgeObjs:
        v1 = np.array(edge.vertex1.coords)
        v2 = np.array(edge.vertex2.coords)
        current_length = np.linalg.norm(v2 - v1)

        if current_length < 0.01 * edge.rest_length:
            print(f"  ⚠️  Edge {edge.id} nearly collapsed!")
            print(f"      Current length: {current_length:.6e}")
            print(f"      Rest length: {edge.rest_length:.6e}")

        if current_length > 2.0 * edge.rest_length:
            print(f"  ⚠️  Edge {edge.id} highly stretched!")
            print(f"      Current length: {current_length:.6e}")
            print(f"      Rest length: {edge.rest_length:.6e}")

    # 6. Check theta bounds
    print("\n6. TWIST ANGLE BOUNDS:")
    for edge in edgeObjs:
        if abs(edge.theta) > 2 * np.pi:
            print(f"  ⚠️  Edge {edge.id} has large theta: {edge.theta:.4f}")
            print(f"      Consider wrapping to [-π, π]")

    print("=" * 50)


def add_theta_wrapping(edgeObjs):
    """
    Wrap all theta values to [-π, π] to prevent unbounded growth.
    Call this after updating edge.theta in your Newton iteration.
    """
    for edge in edgeObjs:
        # Wrap theta to [-π, π]
        edge.theta = np.arctan2(np.sin(edge.theta), np.cos(edge.theta))


def check_reference_frame_continuity(edgeObjs, tolerance=0.1):
    """
    Check if reference directors (a1, a2) have changed significantly.
    Large changes indicate potential instability.
    """
    print("\n=== REFERENCE FRAME CONTINUITY ===")
    for edge in edgeObjs:
        if hasattr(edge, 'a1_old') and hasattr(edge, 'a2_old'):
            a1_dot = np.dot(edge.a1, edge.a1_old)
            a2_dot = np.dot(edge.a2, edge.a2_old)

            # Check orthogonality is maintained
            a1_a2_dot = np.dot(edge.a1, edge.a2)

            if a1_dot < (1 - tolerance) or a2_dot < (1 - tolerance):
                print(f"  ⚠️  Edge {edge.id}: Reference frame jumped!")
                print(f"      a1 continuity: {a1_dot:.4f}")
                print(f"      a2 continuity: {a2_dot:.4f}")

            if abs(a1_a2_dot) > 0.01:
                print(f"  ⚠️  Edge {edge.id}: Directors not orthogonal!")
                print(f"      a1·a2 = {a1_a2_dot:.6f}")


# Usage in your main simulation loop:
"""
for timeStep in range(Nsteps):

    # ... your existing Newton iteration ...

    q_new, u_new = objfun(end_edge, first_end_vertex, edgeObjs, vertexObjs,
                          qOld, uOld, free_index, dt, tol,
                          massVector, massMatrix, EA, EI, GJ, Fg)

    # ADD DIAGNOSTICS HERE:
    if timeStep % 10 == 0:  # Check every 10 steps
        diagnose_sudden_forces(vertexObjs, edgeObjs, Fb, Fs, timeStep)
        check_reference_frame_continuity(edgeObjs)

    # OPTIONAL: Add theta wrapping
    add_theta_wrapping(edgeObjs)

    qOld = q_new.copy()
    uOld = u_new.copy()

    # ... rest of your loop ...
"""