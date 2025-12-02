import numpy as np
# IGNORE THIS FILE - It was used as a quick reference to get a "pretty close" set of vertices and edges.


def CreateHapticNode():
    # ------------------------------------------------------------
    # 1. LEFT PARABOLA (x(y) = 20(y-0.02)^2 - 0.02)
    # ------------------------------------------------------------
    N_parabola = 9
    y_par = np.linspace(5.16e-3, 4.64e-2, N_parabola)  # from bottom to top
    x_par = 20*(y_par - 0.02)**2 - 0.02
    #parabola_pts = np.column_stack((x_par, y_par))
    print(parabola_pts)

    # ------------------------------------------------------------
    # 2. UPPER CIRCLE (x^2 + (y-0.02)^2 = 0.001)
    # ------------------------------------------------------------
    N_circle = 25
    theta_start = -0.55
    theta_end = np.pi + 0.55  # ≈ 4.007 rad
    theta = np.linspace(theta_start, theta_end, N_circle)

    r = 0.0316
    x_circ = r * np.cos(theta)
    y_circ = r * np.sin(theta) + 0.02

    circle_pts = np.column_stack((x_circ, y_circ))

    # ------------------------------------------------------------
    # 3. BOTTOM LINE (y = 0)
    # ------------------------------------------------------------
    N_bottom = 9
    xL = -0.0244949
    xR =  0.0244949
    x_bottom = np.linspace(xL, xR, N_bottom)
    y_bottom = np.zeros_like(x_bottom)
    bottom_pts = np.column_stack((x_bottom, y_bottom))

    # ------------------------------------------------------------
    # COMBINE INTO SINGLE VERTEX ARRAY
    # ------------------------------------------------------------
    vertices = np.vstack((parabola_pts, circle_pts, bottom_pts))

    # ------------------------------------------------------------
    # BUILD EDGE LIST (connect sequentially)
    # ------------------------------------------------------------
    edgeIndex = []
    numP = len(parabola_pts)
    numC = len(circle_pts)
    numB = len(bottom_pts)

    # Parabola edges
    for i in range(numP - 1):
        edgeIndex.append([i, i+1])

    # Circle edges
    offsetC = numP
    for i in range(numC - 1):
        edgeIndex.append([offsetC + i, offsetC + i + 1])

    # Bottom line edges
    offsetB = numP + numC
    for i in range(numB - 1):
        edgeIndex.append([offsetB + i, offsetB + i + 1])

    edgeIndex = np.array(edgeIndex, dtype=int)

    for i, vert in enumerate(vertices):
        print(i, vert)

    #print("vertices:\n", vertices)
    #print("\nedges:\n", edgeIndex)

    return
