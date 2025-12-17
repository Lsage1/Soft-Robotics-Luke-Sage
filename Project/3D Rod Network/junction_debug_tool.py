import numpy as np


def diagnose_junction_forces(vertexObjs, edgeObjs, Fb, q_new):
    """
    Diagnostic tool to understand junction force magnitudes and balance.
    Call this inside your objfun convergence loop.
    """

    print("\n" + "=" * 80)
    print("JUNCTION FORCE DIAGNOSTIC")
    print("=" * 80)

    for v in vertexObjs:
        if not v.junction:
            continue

        print(f"\nJunction at vertex {v.id}:")
        print(f"  Position: {v.coords}")

        # Get the bending forces on this junction vertex
        junction_forces = Fb[v.index]
        print(f"  Bending forces on junction: {junction_forces}")
        print(f"  |Fb| = {np.linalg.norm(junction_forces):.6e}")

        # Check forces on each connected edge
        print(f"  Connected edges: {[e.id for e in v.edges]}")
        for edge in v.edges:
            other_v = edge.get_other_vertex(v)
            edge_forces = Fb[other_v.index]
            print(
                f"    Edge {edge.id} → vertex {other_v.id}: Fb = {edge_forces}, |Fb| = {np.linalg.norm(edge_forces):.6e}")

        # Check twisting forces
        print(f"  Twist forces:")
        for edge in v.edges:
            twist_force = Fb[edge.theta_index]
            print(f"    Edge {edge.id}: Fb_theta = {twist_force:.6e}")

        # Compute current curvatures vs rest curvatures
        print(f"\n  Edge pair curvatures:")
        for i, (e0, e1) in enumerate(v.edgePairs):
            print(f"    Pair {i}: edges ({e0.id}, {e1.id})")

            if i < len(v.junction_rest_kappa):
                kappa_rest = v.junction_rest_kappa[i]
                print(f"      Rest kappa: {kappa_rest}")

            if i < len(v.junction_kappa):
                kappa_current = v.junction_kappa[i]
                print(f"      Current kappa: {kappa_current}")

                if i < len(v.junction_rest_kappa):
                    delta_kappa = kappa_current - kappa_rest
                    print(f"      Δkappa: {delta_kappa}, |Δκ| = {np.linalg.norm(delta_kappa):.6e}")


def check_jacobian_conditioning(J_free, free_index):
    """
    Check if the Jacobian is well-conditioned.
    Poor conditioning can cause convergence issues.
    """

    print("\n" + "=" * 80)
    print("JACOBIAN CONDITIONING CHECK")
    print("=" * 80)

    # Compute condition number
    try:
        cond = np.linalg.cond(J_free)
        print(f"Condition number: {cond:.6e}")

        if cond > 1e10:
            print("⚠️  WARNING: Jacobian is poorly conditioned!")
            print("   This will cause convergence issues.")
        elif cond > 1e6:
            print("⚠️  CAUTION: Jacobian conditioning is marginal.")
        else:
            print("✓ Jacobian is well-conditioned.")

        # Check eigenvalues
        eigvals = np.linalg.eigvals(J_free)
        min_eigval = np.min(np.abs(eigvals))
        max_eigval = np.max(np.abs(eigvals))

        print(f"\nEigenvalue range:")
        print(f"  Min |λ|: {min_eigval:.6e}")
        print(f"  Max |λ|: {max_eigval:.6e}")
        print(f"  Ratio: {max_eigval / min_eigval:.6e}")

        # Check for near-zero or negative eigenvalues
        near_zero = np.sum(np.abs(eigvals) < 1e-10)
        negative = np.sum(eigvals < -1e-10)

        if near_zero > 0:
            print(f"⚠️  WARNING: {near_zero} near-zero eigenvalues!")
            print("   System may be under-constrained or singular.")

        if negative > 0:
            print(f"⚠️  WARNING: {negative} negative eigenvalues!")
            print("   Jacobian should be positive definite but isn't.")

    except np.linalg.LinAlgError:
        print("⚠️  ERROR: Could not compute condition number (Jacobian is singular!)")

    print("=" * 80 + "\n")


def check_force_balance(Fs, Fb, Fg, massVector, dt, qOld, q_new, uOld):
    """
    Check if forces are balanced and reasonable.
    """

    print("\n" + "=" * 80)
    print("FORCE BALANCE CHECK")
    print("=" * 80)

    # Compute inertial force
    Fi = massVector / dt * ((q_new - qOld) / dt - uOld)

    print(f"Force magnitudes:")
    print(f"  Stretching: |Fs| = {np.linalg.norm(Fs):.6e}, max = {np.max(np.abs(Fs)):.6e}")
    print(f"  Bending:    |Fb| = {np.linalg.norm(Fb):.6e}, max = {np.max(np.abs(Fb)):.6e}")
    print(f"  External:   |Fg| = {np.linalg.norm(Fg):.6e}, max = {np.max(np.abs(Fg)):.6e}")
    print(f"  Inertial:   |Fi| = {np.linalg.norm(Fi):.6e}, max = {np.max(np.abs(Fi)):.6e}")

    # Check ratios
    total_elastic = np.linalg.norm(Fs + Fb)
    if total_elastic > 0:
        print(f"\nForce ratios:")
        print(f"  Bending/Stretching: {np.linalg.norm(Fb) / np.linalg.norm(Fs):.6e}")
        print(f"  External/Elastic: {np.linalg.norm(Fg) / total_elastic:.6e}")
        print(f"  Inertial/Elastic: {np.linalg.norm(Fi) / total_elastic:.6e}")

    # Check if any force component is unusually large
    all_forces = Fs + Fb + Fg
    if np.max(np.abs(all_forces)) > 1e6:
        print(f"\n⚠️  WARNING: Very large forces detected!")
        print(f"     Max force component: {np.max(np.abs(all_forces)):.6e}")
        print(f"     This may indicate instability.")

    print("=" * 80 + "\n")

# Usage in objfun:
# Add these calls inside your convergence loop:
#
# if iter == 1 or iter % 10 == 0:  # Check every 10 iterations
#     diagnose_junction_forces(vertexObjs, edgeObjs, Fb, q_new)
#     check_force_balance(Fs, Fb, Fg, massVector, dt, qOld, q_new, uOld)
#
# if iter == 1:  # Check once at start
#     check_jacobian_conditioning(J_free, freeIndex)