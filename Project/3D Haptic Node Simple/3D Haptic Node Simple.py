import numpy as np
import matplotlib.pyplot as plt
from plotrod_simple import plotrod_simple
from computeTangent import computeTangent
from computeSpaceParallel import computeSpaceParallel
from computeMaterialDirectors import computeMaterialDirectors
from getKappa import getKappa
from objfun import objfun
from computeKappa import computeKappa
from objfun_test import objfun_test

verts = [
    np.array([[0,0,0], [0,0.05,0], [0,0.1,-0.05], [0,0.1,-0.10]]),
    np.array([[0,0,0], [0,-0.05,0], [0,-0.1,-0.05], [0,-0.1,-0.10]]),
    np.array([[0,0,0], [0.05,0,0], [0.1,0,-0.05], [0.1,0,-0.10]])
]

nRods = len(verts)

# Pre-allocate arrays per rod
nv     = [len(v)        for v in verts]
ne     = [len(v)-1      for v in verts]
ndof   = [3*nv[r] + ne[r] for r in range(nRods)]  # trimmed last theta

q      = [np.zeros(nd)  for nd in ndof]
qOld   = [None]*nRods
uOld   = [None]*nRods

refLen        = [np.zeros(ne[r]) for r in range(nRods)]
vorRefLen     = [np.zeros(nv[r]) for r in range(nRods)]
tangent       = [None]*nRods
a1            = [None]*nRods
a2            = [None]*nRods
m1            = [None]*nRods
m2            = [None]*nRods
kappaBar      = [None]*nRods
twistBar      = [None]*nRods

# ---------------- MAIN PER-ROD LOOP ----------------
for r in range(nRods):

    # Build q from verts
    for c in range(nv[r]):
        q[r][4*c:4*c+3] = verts[r][c]
        if 4*c+3 < ndof[r]:
            q[r][4*c+3] = 0.0  # theta (skip last if beyond ndof)

    qOld[r] = q[r].copy()
    uOld[r] = np.zeros_like(q[r])

    # Reference lengths
    for c in range(ne[r]):
        refLen[r][c] = np.linalg.norm(verts[r][c+1] - verts[r][c])

    # Voronoi reference lengths
    for c in range(nv[r]):
        if c == 0:
            vorRefLen[r][c] = 0.5 * refLen[r][0]
        elif c == nv[r]-1:
            vorRefLen[r][c] = 0.5 * refLen[r][c-1]
        else:
            vorRefLen[r][c] = 0.5 * (refLen[r][c-1] + refLen[r][c])

    # Tangents
    tangent[r] = computeTangent(qOld[r])

    # Initial a1 direction
    t0 = tangent[r][0]
    arb = np.array([0,0,-1])
    cross = np.cross(t0, arb)
    if np.linalg.norm(cross) < 1e-3:
        arb = np.array([0,1,0])
        cross = np.cross(t0, arb)
    a1_first = cross / np.linalg.norm(cross)

    # Space parallel transport
    a1[r], a2[r] = computeSpaceParallel(a1_first, qOld[r])

    # Theta edges
    theta_nodes = qOld[r][3::4]
    theta_edges = theta_nodes[:ne[r]]

    # Directors
    m1[r], m2[r] = computeMaterialDirectors(a1[r], a2[r], theta_edges)

    # Curvature
    kappaBar[r] = getKappa(qOld[r], m1[r], m2[r])

    # Twist
    twistBar[r] = np.zeros(nv[r])

###################################################################
refLenJunction = 0.05 ######## CHANGE LATER ****


# ---------------- FIXED/FREE DOFS ----------------
fixedIndex = []
freeIndex  = []

for r in range(nRods):
    last_node = nv[r] - 1
    fi = [4 * last_node + i for i in range(3)]  # fix x,y,z of last node
    all_indices = np.arange(ndof[r])
    fr = np.setdiff1d(all_indices, fi)
    fixedIndex.append(fi)
    freeIndex.append(fr)

# ---------------- MASS MATRIX ----------------
rho = 1200
r0 = 0.001
L = np.sum(np.linalg.norm(verts[0][1:] - verts[0][:-1], axis=1))
totalM = L * np.pi * r0**2 * rho
dm = totalM / ne[0]

massVector = np.zeros(ndof[0])
for c in range(nv[0]):
    ind = [4*c, 4*c+1, 4*c+2]
    if c==0 or c==nv[0]-1:
        massVector[ind] = dm/2
    else:
        massVector[ind] = dm
for c in range(ne[0]):
    if 4*c+3 < ndof[0]:
        massVector[4*c+3] = 0.5 * dm * r0**2

massMatrix = np.diag(massVector)

# ---------------- EXTERNAL FORCES ----------------
F_control = [.1, 0, 0]
Fg = [np.zeros(ndof[0]) for _ in range(nRods)]
for r in range(nRods):
    last_node = nv[r]-3
    ind = [4*last_node, 4*last_node+1, 4*last_node+2]
    Fg[r][ind] = [0,0,-F_control[r]]

# ---------------- ELASTIC PARAMETERS ----------------
Y = 7e6
nu = 0.5
G = Y/(2*(1+nu))
EA = Y*np.pi*r0**2
EI = Y*np.pi*r0**4/4
GJ = G*np.pi*r0**4/2
dt = 0.01
totalTime = 1.0
tol = EI / L**2 * 1e-3

# ---------------- REFERENCE TWIST ----------------
refTwist = [np.zeros(nv[r]) for r in range(nRods)]

# ---------------- TIME STEPPING ----------------
Nsteps = round(totalTime/dt)
ctime = 0

a1_old = [a1[r].copy() for r in range(nRods)]
a2_old = [a2[r].copy() for r in range(nRods)]

plotrod_simple(qOld[0], qOld[1], qOld[2], 0)

print("enteringLoop")


for timeStep in range(Nsteps):
    q_new, u_new, a1_new, a2_new = objfun(
        qOld, uOld, a1_old, a2_old,
        freeIndex, dt, tol, refTwist,
        massVector, massMatrix,
        EA, refLen, refLenJunction,
        EI, GJ, vorRefLen,
        kappaBar, twistBar,
        Fg, nRods, tangent, nv[0]
    )

    if timeStep % 10 == 0:
        plotrod_simple(q_new[0], q_new[1], q_new[2], ctime)
        plt.show()

    ctime += dt
    qOld = [q.copy() for q in q_new]
    uOld = [u.copy() for u in u_new]
    a1_old = [a.copy() for a in a1_new]
    a2_old = [a.copy() for a in a2_new]
